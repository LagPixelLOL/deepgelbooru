import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm
import deep_danbooru_model
import pillow_avif
from PIL import Image

TORCH_DTYPE = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_tensor(image_path, use_cuda=True):
    with Image.open(image_path) as pic:
        return torchvision.transforms.functional.pil_to_tensor(pic.convert("RGB").resize((512, 512))).to(device if use_cuda else "cpu", TORCH_DTYPE).permute(1, 2, 0).unsqueeze(0) / 255

class DeepDanbooruDataset(Dataset):

    def __init__(self, images_dir):
        self.data = []
        for path in sorted(os.listdir(images_dir)):
            if path.endswith(".txt"):
                continue
            path = os.path.join(images_dir, path)
            if not os.path.isfile(path):
                continue
            tags_path = os.path.splitext(path)[0] + ".txt"
            if not os.path.isfile(tags_path):
                continue
            self.data.append((path, tags_path))
        self.label_mapping = {label: index for index, label in enumerate(model.tags)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = get_image_tensor(self.data[idx][0], False)

        with open(self.data[idx][1], "r", encoding="utf8") as file:
            tags_text = file.read()
        labels = []
        for tag in tags_text.split(","):
            tag = tag.strip()
            if tag:
                tag = tag.replace(" ", "_")
                if tag == "nsfw": tag = "rating:explicit"
                elif tag == "qfw": tag = "rating:questionable"
                elif tag == "sfw": tag = "rating:safe"
                labels.append(tag)

        label_tensor = torch.zeros(1, len(model.tags), dtype=TORCH_DTYPE)
        for label in labels:
            label_idx = self.label_mapping.get(label)
            if label_idx is None:
                continue
            label_tensor[0, label_idx] = 1

        return image, label_tensor

def find_last():
    highest_epoch = 0
    if os.path.isdir("train_results"):
        for file in os.listdir("train_results"):
            if file.startswith("model_epoch_"):
                num_epoch = int(os.path.splitext(file)[0][12:])
                if num_epoch > highest_epoch:
                    highest_epoch = num_epoch
    if highest_epoch > 0:
        optim_path = os.path.join("train_results", f"optim_epoch_{highest_epoch}.bin")
        if os.path.isfile(optim_path):
            optim_sd = torch.load(optim_path, device, weights_only=False)
        else:
            optim_sd = None
        return torch.load(os.path.join("train_results", f"model_epoch_{highest_epoch}.bin"), device, weights_only=False), optim_sd, highest_epoch
    return None, None, highest_epoch

model_sd, optim_sd, highest_epoch = find_last()

if model_sd is not None:
    print(f"Found previous training at epoch {highest_epoch}, resuming...")
else:
    model_sd = "deepdanbooru.bin"
model = deep_danbooru_model.DeepDanbooruModel.from_single_file(model_sd, device, TORCH_DTYPE)

batch_size = 128
dataset = DeepDanbooruDataset("/root/anime-collection/images")
train_dataset, eval_dataset = random_split(dataset, [0.99, 0.01], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), generator=torch.Generator().manual_seed(42))
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), generator=torch.Generator().manual_seed(42))

criterion = nn.BCELoss()
learning_rate = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
if optim_sd:
    print("Found previous training optimizer state too, resuming...")
    optimizer.load_state_dict(optim_sd)
for group in optimizer.param_groups:
    group["lr"] = learning_rate
scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)

del model_sd, optim_sd

@torch.no_grad
def test():
    model.eval()
    torch.cuda.empty_cache()
    if not os.path.isdir("samples"):
        return
    image_files = os.listdir("samples")
    if len(image_files) <= 0:
        return
    images_tensor = torch.tensor([], device=device, dtype=TORCH_DTYPE)
    for file in image_files:
        images_tensor = torch.cat((images_tensor, get_image_tensor(os.path.join("samples", file))), 0)
    results = model(images_tensor)
    print()
    for file, result in zip(image_files, results):
        print(f"Sample \"{file}\": ", end="", flush=True)
        first = True
        for i, prob in enumerate(result):
            if prob >= 0.5:
                if first:
                    first = False
                else:
                    print(", ", end="", flush=True)
                print(model.tags[i].replace("_", " "), end="", flush=True)
        print("\n")

@torch.no_grad
def evaluate():
    model.eval()
    torch.cuda.empty_cache()
    eval_loss = 0.0
    eval_correct_labels = 0
    eval_sample_labels = 0

    for images, labels in tqdm.tqdm(eval_dataloader):
        images = images.squeeze(1).to(device)
        labels = labels.squeeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        eval_loss += loss.item()

        predicted_labels = outputs > 0.5
        eval_correct_labels += int(torch.logical_and(predicted_labels, labels).sum())
        eval_sample_labels += int(labels.sum())

    eval_acc = eval_correct_labels / eval_sample_labels
    eval_loss /= len(eval_dataloader)
    print(f"Eval Loss: {eval_loss:.5g}, Eval Accuracy: {eval_acc:.5g}")

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

num_epochs = 69

def main():
    test()
    evaluate()
    for epoch in range(highest_epoch, num_epochs):
        model.train()
        torch.cuda.empty_cache()
        running_loss = 0.0
        running_correct_labels = 0
        running_sample_labels = 0
        step_count = 0
        for images, labels in tqdm.tqdm(train_dataloader):
            step_count += 1
            images = images.squeeze(1).to(device)
            labels = labels.squeeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            predicted_labels = outputs > 0.5
            running_correct_labels += int(torch.logical_and(predicted_labels, labels).sum())
            running_sample_labels += int(labels.sum())

            if step_count % 50 == 0:
                step_acc = running_correct_labels / running_sample_labels
                step_loss = running_loss / step_count
                tqdm.tqdm.write(f"Loss: {step_loss:.5g}, Accuracy: {step_acc:.5g}, LR: {scheduler.get_last_lr()[0]:.5g}")

        epoch_acc = running_correct_labels / running_sample_labels
        epoch_loss = running_loss / step_count
        print(f"Finished epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.5g}, Accuracy: {epoch_acc:.5g}")

        print("Saving model and optimizer states...")
        os.makedirs("train_results", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("train_results", f"model_epoch_{epoch + 1}.bin"))
        torch.save(optimizer.state_dict(), os.path.join("train_results", f"optim_epoch_{epoch + 1}.bin"))
        print("Saved.")

        test()
        evaluate()

if __name__ == "__main__":
    main()
