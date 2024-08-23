import os
import random
import argparse
import concurrent.futures
import torch
import torchvision
import deep_danbooru_model
from PIL import Image

TORCH_DTYPE = torch.bfloat16
MIN_PROB = 0.5

@torch.no_grad
def get_tags(images_tensor, model):
    result = model(images_tensor)
    ret = []
    for e in result:
        tags = []
        for i, prob in enumerate(e):
            if prob >= MIN_PROB:
                tag = model.tags[i]
                if tag.startswith("rating"):
                    continue
                tag = tag.replace("_", " ")
                tags.append(tag)
        ret.append(tags)
    return ret

def get_image_tensor(image_path):
    with Image.open(image_path) as pic:
        return torchvision.transforms.functional.pil_to_tensor(pic.convert("RGB").resize((512, 512))).to("cuda", TORCH_DTYPE).permute(1, 2, 0).unsqueeze(0) / 255

def parse_args():
    parser = argparse.ArgumentParser(description="Tag images using some AI resnet stuffs.")
    parser.add_argument("-i", "--input", type=str, required=True, help="The folder containing images")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="The batch size when tagging the images")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Loading model...")
    model = deep_danbooru_model.DeepDanbooruModel.from_single_file("deepdanbooru.bin", "cuda", TORCH_DTYPE)
    print("Getting image paths and filtering unneeded images...")
    image_files = []
    for image_file in [os.path.join(args.input, f) for f in os.listdir(args.input) if not f.endswith(".txt")]:
        if not os.path.isfile(image_file):
            continue
        tags_file = os.path.splitext(image_file)[0] + ".txt"
        if not os.path.isfile(tags_file):
            image_files.append(image_file)
            continue
        with open(tags_file, "r", encoding="utf8") as file:
            tags_text = file.read()
        if len(tags_text) > 256:
            continue
        image_files.append(image_file)
    image_files.sort()
    print(f"Processing {len(image_files)} images...")
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size)
    for i in range(0, len(image_files), args.batch_size):
        print(f"At batch #{i // args.batch_size + 1}...")
        chunk = image_files[i:i + args.batch_size]
        images_tensor = torch.tensor([], device="cuda", dtype=TORCH_DTYPE)
        futures = []
        for image_file in chunk:
            futures.append(thread_pool.submit(get_image_tensor, image_file))
        for future in futures:
            images_tensor = torch.cat((images_tensor, future.result()), 0)
        chunk_tags = get_tags(images_tensor, model)
        for i, image_file in enumerate(chunk):
            tags_file = os.path.splitext(image_file)[0] + ".txt"
            image_tags = set()
            if os.path.isfile(tags_file):
                with open(tags_file, "r", encoding="utf8") as file:
                    tags_text = file.read()
                for tag in tags_text.split(","):
                    tag = tag.strip()
                    if tag:
                        image_tags.add(tag)
            image_tags.update(chunk_tags[i])
            image_tags = list(image_tags)
            random.shuffle(image_tags)
            tags_text = ", ".join(image_tags)
            with open(tags_file, "w", encoding="utf8") as file:
                file.write(tags_text)

if __name__ == "__main__":
    main()
