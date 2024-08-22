import time
import torch
import torchvision
import deep_danbooru_model
from PIL import Image

torch.set_grad_enabled(False)

with torch.device("meta"):
    model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load("deepdanbooru.bin", "cuda", weights_only=False), assign=True, strict=True)

pic = Image.open("test.png").convert("RGB").resize((512, 512))
x = torchvision.transforms.functional.pil_to_tensor(pic).to("cuda", torch.bfloat16).permute(1, 2, 0).unsqueeze(0) / 255

r = model(x)
y = r[0]

tags = []
for i, p in enumerate(y):
    p = float(p)
    if p >= 0.5:
        tag = model.tags[i]
        if tag.startswith("rating"):
            continue
        tag = tag.replace("_", " ")
        tags.append(tag)
        print(tag, p)
tags_text = ", ".join(tags)

with open("result.txt", "w", encoding="utf8") as file:
    file.write(tags_text)
