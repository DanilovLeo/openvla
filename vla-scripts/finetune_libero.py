"""
finetune_libero.py

LoRA fine-tuning of OpenVLA on LIBERO HDF5 demonstrations filtered to keyframes
by a configurable extractor.  Drop-in replacement for finetune.py; same training
loop, same W&B logging, but replaces the RLDS/TFDS pipeline with a plain PyTorch
Dataset that reads directly from HDF5.

Run (single GPU):
    python vla-scripts/finetune_libero.py \
        --hdf5_dir /path/to/libero_spatial \
        --extractor_name velocity_zero \
        --velocity_percentile 25 \
        --run_root_dir runs/

Run (multi-GPU):
    torchrun --standalone --nnodes 1 --nproc-per-node 2 \
        vla-scripts/finetune_libero.py \
        --hdf5_dir /path/to/libero_spatial \
        --task_filter "black_bowl_from_table_center,black_bowl_next_to_plate" \
        --extractor_name uniform \
        --run_root_dir runs/

--task_filter is a comma-separated list of substrings; a file is included if any
  substring appears in its stem.  Leave empty to use all files in --hdf5_dir.
"""

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.libero_dataset import LiberoKeyframeDataset, build_extractor
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure keyframe-selector/src is importable (default repo layout)
_kf_src = Path(__file__).parent.parent.parent / "keyframe-selector" / "src"
if _kf_src.exists() and str(_kf_src) not in sys.path:
    sys.path.insert(0, str(_kf_src))


@dataclass
class LiberoFinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"

    # --- LIBERO data ---
    hdf5_dir: Path = Path(".")                    # Directory containing .hdf5 demo files
    task_filter: str = ""                         # Comma-separated substrings; empty = all files
    keyframe_selector_src: str = ""               # Override path to keyframe-selector/src (if needed)

    # --- Extractor ---
    extractor_name: str = "uniform"               # uniform | velocity_zero | gripper_state | awe
    uniform_n_keyframes: int = 10
    velocity_percentile: float = 25.0
    velocity_min_dist: int = 5
    gripper_min_dist: int = 5
    awe_error_threshold: float = 0.01

    # --- Output ---
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # --- Training ---
    batch_size: int = 16
    max_steps: int = 50_000
    save_steps: int = 5_000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    save_latest_checkpoint_only: bool = True

    # --- LoRA ---
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    # --- W&B ---
    wandb_project: str = "openvla"
    wandb_entity: str = "your-entity"
    run_id_note: Optional[str] = None
    # fmt: on


def resolve_hdf5_paths(hdf5_dir: Path, task_filter: str) -> List[str]:
    """Return sorted HDF5 paths from hdf5_dir, filtered by task_filter substrings."""
    all_files = sorted(hdf5_dir.expanduser().glob("*.hdf5"))
    if not all_files:
        raise FileNotFoundError(f"No .hdf5 files found in {hdf5_dir}")
    if not task_filter:
        return [str(p) for p in all_files]
    filters = [f.strip() for f in task_filter.split(",") if f.strip()]
    selected = [p for p in all_files if any(f in p.stem for f in filters)]
    if not selected:
        raise ValueError(
            f"task_filter={task_filter!r} matched no files in {hdf5_dir}.\n"
            f"Available stems: {[p.stem for p in all_files]}"
        )
    return [str(p) for p in selected]


@draccus.wrap()
def finetune(cfg: LiberoFinetuneConfig) -> None:
    # Optionally override keyframe-selector src path
    if cfg.keyframe_selector_src:
        kf_src = Path(cfg.keyframe_selector_src).expanduser()
        if str(kf_src) not in sys.path:
            sys.path.insert(0, str(kf_src))

    hdf5_paths = resolve_hdf5_paths(cfg.hdf5_dir, cfg.task_filter)
    extractor, traj_key = build_extractor(
        cfg.extractor_name,
        uniform_n=cfg.uniform_n_keyframes,
        velocity_percentile=cfg.velocity_percentile,
        velocity_min_dist=cfg.velocity_min_dist,
        gripper_min_dist=cfg.gripper_min_dist,
        awe_eps=cfg.awe_error_threshold,
    )

    print(
        f"Fine-tuning OpenVLA `{cfg.vla_path}` on {len(hdf5_paths)} LIBERO task(s) "
        f"with extractor={extractor.name}"
    )
    for p in hdf5_paths:
        print(f"  {p}")

    assert torch.cuda.is_available(), "Fine-tuning requires at least one GPU!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}"
        f"+libero_{extractor.name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    prompt_builder_fn = (
        PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder
    )

    # Build dataset — this pre-loads all keyframes into memory
    vla_dataset = LiberoKeyframeDataset(
        hdf5_paths=hdf5_paths,
        extractor=extractor,
        traj_key=traj_key,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
        dataset_name=f"libero_{extractor.name}",
    )
    print(vla_dataset.summary())

    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    recent_losses             = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies  = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses          = deque(maxlen=cfg.grad_accumulation_steps)

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        batch_idx = 0
        # Loop over epochs implicitly (dataloader cycles)
        while True:
            for batch in dataloader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss

                (loss / cfg.grad_accumulation_steps).backward()

                action_logits = output.logits[
                    :, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1
                ]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(
                    continuous_actions_pred, continuous_actions_gt
                )

                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss.item())

                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()

                    if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                        wandb.log(
                            {
                                "train_loss":      sum(recent_losses) / len(recent_losses),
                                "action_accuracy": sum(recent_action_accuracies) / len(recent_action_accuracies),
                                "l1_loss":         sum(recent_l1_losses) / len(recent_l1_losses),
                            },
                            step=gradient_step_idx,
                        )

                    if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                        if distributed_state.is_main_process:
                            print(f"Saving checkpoint at step {gradient_step_idx}")
                            save_dir = adapter_dir if cfg.use_lora else run_dir
                            processor.save_pretrained(run_dir)
                            vla.module.save_pretrained(save_dir)
                        dist.barrier()

                        if cfg.use_lora:
                            base_vla = AutoModelForVision2Seq.from_pretrained(
                                cfg.vla_path,
                                torch_dtype=torch.bfloat16,
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                            )
                            merged = PeftModel.from_pretrained(base_vla, adapter_dir)
                            merged = merged.merge_and_unload()
                            if distributed_state.is_main_process:
                                if cfg.save_latest_checkpoint_only:
                                    merged.save_pretrained(run_dir)
                                else:
                                    ckpt_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                                    os.makedirs(ckpt_dir, exist_ok=True)
                                    save_dataset_statistics(vla_dataset.dataset_statistics, ckpt_dir)
                                    processor.save_pretrained(ckpt_dir)
                                    merged.save_pretrained(ckpt_dir)
                        dist.barrier()

                    if gradient_step_idx >= cfg.max_steps:
                        print(f"Reached max_steps={cfg.max_steps}. Done.")
                        return

                batch_idx += 1


if __name__ == "__main__":
    finetune()
