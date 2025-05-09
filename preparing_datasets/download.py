"""
Downloads all the necessary datasets for LLaVA training steps 1 and 2 as
explained in LLaVA's README file.

To run it, move this script and the OCR-VQA_dataset.json to the same directory
and run it.

Final folder structure once succeeded:

/llava-datasets
├── pretrain_feature_alignment
│   └── (files from huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
└── visual_instruction_tuning
    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── ocr_vqa
    │   └── images
    ├── textvqa
    │   └── train_images
    └── vg
        ├── VG_100K
        └── VG_100K_2

IMPORTANT NOTES:
- for me, some timeouts made me download some files manually and copy over (see below)
- for the OCR-VQA dataset, there are some more caveats (see below)
"""

import os
import time
import requests
import zipfile
import shutil

####################################################################################################
# Setup
ROOT = "/llava-datasets"
PFA_ROOT = os.path.join(ROOT, "pretrain_feature_alignment")
VIT_ROOT = os.path.join(ROOT, "visual_instruction_tuning")


####################################################################################################
# Utils
def download_file(url, output_path, timeout=60, retries=3, backoff_factor=5):
    for attempt in range(retries):
        try:
            print(f"Downloading {url} (attempt {attempt + 1})")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            print("Done.")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2**attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)
    print("Unzipped and removed archive.")


# Create folder structure
dirs = {
    "pfa": PFA_ROOT,
    "coco": os.path.join(VIT_ROOT, "coco/train2017"),
    "gqa": os.path.join(VIT_ROOT, "gqa/images"),
    "ocr_vqa": os.path.join(VIT_ROOT, "ocr_vqa/images"),
    "textvqa": os.path.join(VIT_ROOT, "textvqa/train_images"),
    "vg1": os.path.join(VIT_ROOT, "vg/VG_100K"),
    "vg2": os.path.join(VIT_ROOT, "vg/VG_100K_2"),
}
for path in dirs.values():
    os.makedirs(path, exist_ok=True)

####################################################################################################
# Pretrain alignment dataset (~27GB)
print("Downloading pretrain feature alignment data...")
download_file(
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json",
    os.path.join(PFA_ROOT, "blip_laion_cc_sbu_558k.json"),
)
download_file(
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k_meta.json",
    os.path.join(PFA_ROOT, "blip_laion_cc_sbu_558k_meta.json"),
)
download_file(
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip",
    os.path.join(PFA_ROOT, "images.zip"),
)
unzip_file(
    os.path.join(PFA_ROOT, "images.zip"), PFA_ROOT
)  # NOTE: Not sure atm where this unzipped stuff has to be
print("Done.")

####################################################################################################
# Visual Instruction Tuning datasets
print("Downloading Visual Instruction Tuning datasets...")

# LLaVA-Instruct JSON (~1GB)
download_file(
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json",
    os.path.join(VIT_ROOT, "llava_v1_5_mix665k.json"),
)

# COCO (~19GB)
coco_zip = os.path.join(VIT_ROOT, "coco/train2017.zip")
download_file("http://images.cocodataset.org/zips/train2017.zip", coco_zip)
unzip_file(coco_zip, os.path.join(VIT_ROOT, "coco"))

###
# GQA (~21GB)
# gqa_zip = os.path.join(VIT_ROOT, "gqa/images.zip")
# download_file("https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip", gqa_zip)
# unzip_file(gqa_zip, dirs["gqa"])
# -> Times out; downloaded it manually, copied it to the cluster, and unzipped it there
# (That is, once I had the images.zip in /llava-datasets/visual_instruction_tuning/gqa, I simply
#  ran `unzip images.zip -d images` in that directory)
# NOTE: Make sure its not an images folder contained within another images folder
###

# OCR-VQA -- Using the information (including dataset.json) from
#  https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing
# (~9GB)
print("Downloading OCR-VQA images using OCR-VQA_dataset.json...")

import json
import urllib.request

ocr_json_path = os.path.join(ROOT, "OCR-VQA_dataset.json")

# Load dataset and download images (NOTE: this takes quite a while; could probably paralleize)
with open(ocr_json_path, "r") as f:
    ocr_data = json.load(f)

for k, entry in ocr_data.items():
    ext = os.path.splitext(entry["imageURL"])[1]
    out_path = os.path.join(dirs["ocr_vqa"], f"{k}{ext}")
    try:
        print(f"Downloading OCR-VQA image: {entry['imageURL']}")
        urllib.request.urlretrieve(entry["imageURL"], out_path)
    except Exception as e:
        print(f"Failed to download {entry['imageURL']}: {e}")

# NOTE: I forgot to save as .jpg; you can fix this above or run
#    for f in *.gif; do mv "$f" "${f%.gif}.jpg"; done
#  and
#    for f in *.png; do mv "$f" "${f%.png}.jpg"; done
#  afterwards.
# NOTE: I encountered
#    FileNotFoundError: [Errno 2] No such file or directory: '/llava-datasets/visual_instruction_tuning/ocr_vqa/images/689852649.jpg'
#  and looking through the `OCR-VQA_dataset.json` file, I actually found that the link
#    http://ecx.images-amazon.com/images/I/61955GMME8L.jpg
#  to that image leads to "Not found". So what I did was to actually remove this
#   image from the llava_v1_5_mix665k.json file.

# TextVQA (~7GB)
textvqa_zip = os.path.join(VIT_ROOT, "textvqa/train_val_images.zip")
download_file(
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip", textvqa_zip
)
unzip_file(textvqa_zip, os.path.join(VIT_ROOT, "textvqa"))

###
# Below, sadly, the same as for GQA...
# # VG part 1 (~10GB)
# vg1_zip = os.path.join(VIT_ROOT, "vg/VG_100K.zip")
# download_file("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", vg1_zip)
# unzip_file(vg1_zip, dirs["vg1"])
# NOTE: The images have to be directly in vg/VG_100K, not in vg/VG_100K/images

# # VG part 2 (~5GB)
# vg2_zip = os.path.join(VIT_ROOT, "vg/VG_100K_2.zip")
# download_file("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", vg2_zip)
# unzip_file(vg2_zip, dirs["vg2"])
# NOTE: The images have to be directly in vg/VG_100K_2, not in vg/VG_100K_2/images
###

print("All downloads and extractions completed.")
