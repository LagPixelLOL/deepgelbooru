import os
import time
import torch
import torchvision
import deep_danbooru_model
from PIL import Image

torch.set_grad_enabled(False)

TORCH_DTYPE = torch.bfloat16

def get_model():
    highest_epoch = 0
    if os.path.isdir("train_results"):
        for file in os.listdir("train_results"):
            if file.startswith("model_epoch_"):
                num_epoch = int(os.path.splitext(file)[0][12:])
                if num_epoch > highest_epoch:
                    highest_epoch = num_epoch
    if highest_epoch > 0:
        return deep_danbooru_model.DeepDanbooruModel.from_single_file(os.path.join("train_results", f"model_epoch_{highest_epoch}.bin"), "cuda", TORCH_DTYPE)
    return deep_danbooru_model.DeepDanbooruModel.from_single_file("deepdanbooru.bin", "cuda", TORCH_DTYPE)

model = get_model()

pic = Image.open("test.jpg").convert("RGB").resize((512, 512))
x = torchvision.transforms.functional.pil_to_tensor(pic).to("cuda", TORCH_DTYPE).permute(1, 2, 0).unsqueeze(0) / 255

r = model(x)
y = r[0]

for i, prob in sorted(((i, float(prob)) for i, prob in enumerate(y)), key=lambda x: x[1]):
    print(model.tags[i], "-", prob)
