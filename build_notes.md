# Notes about how I built stuff, got it to training, and did inference
Goal is to train LLaVA models with other vision encoders than the ones available by default.
Especially, we want to check the performance with [this vision encoder](https://huggingface.co/UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B).
Work for adapting to [OpenCLIP](https://github.com/mlfoundations/open_clip) encoders is inspired by [this](https://gist.github.com/TommyIX/681ad23947c3aa7c8482f4d39849df7d) and [this](https://github.com/haotian-liu/LLaVA/pull/966/commits/f7ad580555a0eee034e56fe9570aa23308bb6eee#diff-37265e6713ed9c53be498990d401380684d625f6fb9be6cc80772c277153fa37)

## Data
See `preparing_datasets/training/download.py` to download and prepare all the datasets necessary to train LLaVA models.

## Docker
I ran `export DOCKER_DEFAULT_PLATFORM=linux/amd64` and built via `cog build -t registry.datexis.com/.../llava-pretrain:latest`.
I could not get it to build without some version adjustments (see `cog.yaml`) and removing flash attention (see `llava/train/train_mem.py`).
But still not all trouble with building for the correct platform with `cog` could be solved with this.
What worked for me as well was to build it first with `cog` (and it will fail), then run `cog debug  > Dockerfile`, then potentially even change this Dockerfile and its `requirements.txt` in the generated `.cog` folder to your needs (e.g., other CUDA and torch version), and finally run `docker build -t registry.datexis.com/.../llava-train:latest . --platform=linux/amd64` with these.

## Pretraining
I did the pretraining on 8xA100 (40GB) using CUDA 11.8 for both the reproducing CLIP and our custom OpenCLIP model.

## Finetuning
For the finetuning, I adjusted the zero3.json config file by disabling `fp16` and enabling `bf16`. This sped up things on the 8xB200 (180GB) with CUDA 12.8 (though used a bit more RAM; it was 90GB per card). This worked for CLIP, but not for our custom OpenCLIP.
For OpenCLIP, I used the zero2.json script with CUDA 11.8 on the 8xA100s again, also opting for bf16 here. My custom OpenCLIP implementation seems not to be nicely comptabile with deepseed stage 3 optimization.
So, dadly, with only 8xA100 (40GB) and zero2 training, I had to go down to a batch size of 2 and the training took 14 hours. Couldn't do it with the B200s, because they are broken. Also, note that on the A100s with CUDA 12.8 the zero2 training did not work (got some weird hardware errors).
Could try incorporating zero3 compatibilty into OpenCLIP stuff to make it work! See https://github.com/baaivision/EVA/blob/master/EVA-CLIP-18B/shinji/eva_clip/factory.py#L168 ... probably not so straight forward though. Idk.
