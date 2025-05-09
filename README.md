# LLaVA with OpenCLIP Vision Encoders

This is a fork of [LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA) that adds compatibility with alternative vision encoders from [OpenCLIP](https://github.com/mlfoundations/open_clip).
If anything remains unclear here, please have a look at these repositories.

## What's New in This Fork

This fork adds support for:
- Using alternative vision encoders with LLaVA, particularly from OpenCLIP
- Testing with the [UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B](https://huggingface.co/UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B) encoder
- Compatibility with different CUDA versions and hardware setups

## Data Preparation

For downloading and preparing all necessary datasets to train LLaVA models, refer to the script in `preparing_datasets/download.py`. The script handles:
- Pretrain feature alignment datasets
- Visual instruction tuning datasets (COCO, GQA, OCR-VQA, TextVQA, VisualGenome)

See comments in the script for handling timeouts and special cases with certain datasets.

## Environment Setup

### Docker

For building with Docker:
```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
cog build -t registry.datexis.com/jwesterhoff/llava-pretrain:latest
```

Note that compared to the original, I could not get it to build without some version adjustments in the `cog.yaml` files and removing flash attention (see `llava/train/train_mem.py`).

If the build fails:
1. Run `cog debug > Dockerfile`
2. Modify the `Dockerfile` and `requirements.txt` in the `.cog` folder as needed (e.g., other CUDA and torch version)
3. Build with `docker build -t registry.datexis.com/jwesterhoff/llava-train:latest . --platform=linux/amd64`

## Training

### Pretraining
- Hardware used: 8x A100 (40GB) with CUDA 11.8
- Compatible with both standard CLIP and custom OpenCLIP models

### Finetuning
- For CLIP models: Used zero3.json config with bf16 on 8x B200 GPUs with CUDA 12.8
- For OpenCLIP models: Used zero2.json with CUDA 11.8 on 8x A100s with bf16
  - Note: Custom OpenCLIP implementation has compatibility issues with DeepSpeed stage 3 optimization, it seems.
  - With zero2 training on 8x A100 (40GB), batch size had to be reduced to 2 (training time: ~14 hours)

### OpenCLIP Integration Notes
- For better DeepSpeed stage 3 compatibility, improvements could be made following approaches like [EVA-CLIP](https://github.com/baaivision/EVA/blob/master/EVA-CLIP-18B/shinji/eva_clip/factory.py#L168)

## Inference

For inference with the trained models, refer to [SCAM project](https://github.com/Bliss-e-V/SCAM).

## Original LLaVA Project

For the complete documentation of the base LLaVA project, including installation instructions, model zoo, evaluation, and more, please refer to the [original LLaVA repository](https://github.com/haotian-liu/LLaVA).

## License

This project follows the licensing terms of the original LLaVA repository.
