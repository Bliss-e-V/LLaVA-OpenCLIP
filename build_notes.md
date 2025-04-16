# Notes about how I build stuff and got it running

## Data
See preparing_datasets/training/download.py to download and prepare all the datasets necessary to train LLaVA models.

## Docker
Goal is to train LLaVA models with other vision encoders than the ones available by default. Especially, we want to check the performance with [this vision encoder](https://huggingface.co/UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B).
Work for adapting to [OpenCLIP](https://github.com/mlfoundations/open_clip) encoders is inspired by [this](https://gist.github.com/TommyIX/681ad23947c3aa7c8482f4d39849df7d) and [this](https://github.com/haotian-liu/LLaVA/pull/966/commits/f7ad580555a0eee034e56fe9570aa23308bb6eee#diff-37265e6713ed9c53be498990d401380684d625f6fb9be6cc80772c277153fa37)

I ran `export DOCKER_DEFAULT_PLATFORM=linux/amd64` and built via `cog build -t registry.datexis.com/jwesterhoff/llava-pretrain:latest`.

NOTE: I could not get it to build without some version adjustments (see cog.yaml) and removing flash attention (see llava/train/train_mem.py).

Not all trouble with building for the correct platform with `cog` could be solved with this.
What worked for me as well was to build it first with `cog` (and it will fail), then run `cog debug  > Dockerfile`, then potentially even change this Dockerfile and its requirements.txt in the generated .cog folder to your needs (e.g., other CUDA and torch version), and finnaly run `docker build -t registry.datexis.com/jwesterhoff/llava-train:latest . --platform=linux/amd64` with that.
