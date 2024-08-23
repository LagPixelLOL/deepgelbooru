import sys
import torch
import deep_danbooru_model

def main():
    tags_to_add = sys.argv[1:]
    if not tags_to_add:
        print("Usage: python", sys.argv[0], "tag_1 tag_2 tag_3 ...")
        sys.exit(1)
    print("Starting...")
    model = deep_danbooru_model.DeepDanbooruModel.from_single_file("deepdanbooru.bin")
    existing_tags = set(model.tags)
    tags_finalized = []
    for tag in tags_to_add:
        tag = tag.strip().replace(" ", "_").lower()
        if tag in existing_tags:
            continue
        existing_tags.add(tag)
        tags_finalized.append(tag)
    if len(tags_finalized) <= 0:
        print("Nothing to add.")
        sys.exit(1)
    print("New tags:", tags_finalized)
    last_layer_param = model.n_Conv_178.weight
    new_tensor = torch.randn(len(tags_finalized), *last_layer_param.shape[1:], dtype=last_layer_param.dtype)
    last_layer_param.data = torch.cat((last_layer_param.data, new_tensor), 0)
    model.tags += tags_finalized
    torch.save(model.state_dict(), "deepdanbooru_added_tags.bin")
    with open("model_tags.txt", "w", encoding="utf8") as file:
        for i, tag in enumerate(model.tags):
            file.write(f"{i} {tag}\n")
    print("Finished, remember to fine-tune the model for the new tags.")

if __name__ == "__main__":
    main()
