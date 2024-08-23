import time
import torch
import torchvision
import deep_danbooru_model
from PIL import Image

torch.set_grad_enabled(False)

TORCH_DTYPE = torch.bfloat16

model = deep_danbooru_model.DeepDanbooruModel.from_single_file("deepdanbooru.bin", "cuda", TORCH_DTYPE)

pic = Image.open("test.jpg").convert("RGB").resize((512, 512))
x = torchvision.transforms.functional.pil_to_tensor(pic).to("cuda", TORCH_DTYPE).permute(1, 2, 0).unsqueeze(0) / 255

r = model(x)
y = r[0]

first = True
for i, prob in enumerate(y):
    if prob >= 0.5:
        if first:
            first = False
        else:
            print(", ", end="", flush=True)
        print(model.tags[i].replace("_", " "), end="", flush=True)
print()
