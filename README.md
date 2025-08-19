# Flux LoRA Training (FP32)

This repository contains training scripts for LoRA on Flux models.  
The setup below describes how we currently run training (previously on RunPod) so it can be ported to fal.ai.

---

## Folder Structure

```
/workspace/
  downloads/               # Pretrained weights
    flux1-dev.safetensors
    t5xxl_fp16.safetensors
    clip_l.safetensors
    ae.safetensors

  projects/
    [MODEL_NAME]/
      configs/
        [MODEL_NAME]-dataset.toml
      train/
        # dataset images + captions (.txt)
      output/
        # trained LoRA checkpoints saved here

  sd-scripts/              # Training scripts (this repo)
```

---

## Example Dataset Config (`[MODEL_NAME]-dataset.toml`)

```toml
[general]
flip_aug = false
color_aug = false
keep_tokens_separator= "|||"
shuffle_caption = false
caption_tag_dropout_rate = 0
caption_extension = ".txt"

[[datasets]]
batch_size = 4
enable_bucket = true
resolution = [1024, 1024]

  [[datasets.subsets]]
  image_dir = "/workspace/projects/[MODEL NAME]/train"
  num_repeats = 10
  #class_tokens = "rileyreidai"
```

---

## Installation

From inside `sd-scripts`:

```bash
pip install -r requirements.txt
```

---

## Training Command

We run training using `accelerate`.  
Example command (multi-GPU, FP32):

```bash
accelerate launch   --num_cpu_threads_per_process 4   --multi_gpu   --num_processes [NUM_GPUS]   flux_train_network.py   --pretrained_model_name_or_path /workspace/downloads/flux1-dev.safetensors   --clip_l /workspace/downloads/clip_l.safetensors   --t5xxl /workspace/downloads/t5xxl_fp16.safetensors   --ae /workspace/downloads/ae.safetensors   --save_model_as safetensors   --seed 486586   --sdpa   --persistent_data_loader_workers   --max_data_loader_n_workers 2   --network_module networks.lora_flux   --network_dim 32   --network_alpha 64   --gradient_checkpointing   --mixed_precision no   --save_precision float   --dataset_config /workspace/projects/[MODEL NAME]/configs/[MODEL NAME]-dataset.toml   --output_dir /workspace/projects/[MODEL NAME]/output   --output_name [MODEL NAME]-flux_lora-fp32_vR1a   --learning_rate 1e-4   --max_train_epochs 40   --t5xxl_max_token_length 512   --disable_mmap_load_safetensors   --cache_text_encoder_outputs_to_disk   --save_every_n_epochs 10   --optimizer_type bitsandbytes.optim.PagedAdEMAMix8bit   --lr_scheduler cosine_with_restarts   --max_bucket_reso 1536   --max_grad_norm 0.0   --timestep_sampling sigmoid   --discrete_flow_shift 3.1582   --model_prediction_type raw   --guidance_scale 1.0
```

---

## Notes for fal.ai Team

- Current training is done with **fp32** (`--mixed_precision no`, `--save_precision float`).
- Default fal.ai pipelines use bf16, but we require fp32 for quality reasons.
- Expected to use multiple GPUs (`--multi_gpu`).
- Datasets and pretrained weights are mounted under `/workspace/`.
