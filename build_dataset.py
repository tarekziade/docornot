import os
from PIL import Image
from datasets import (
    Dataset,
    Features,
    ClassLabel,
    Image as DatasetsImage,
    DatasetDict,
    dataset_dict,
)

MAX_PER_SOURCE = 8000

FLICKR8K_DIR = "./flickr8k/Images"
RVL_CDIP = "./rvl-cdip"


def get_images():
    count = 0

    # Processing pictures
    for file in os.listdir(FLICKR8K_DIR):
        if file.endswith(".jpg"):
            yield {
                "image": Image.open(os.path.join(FLICKR8K_DIR, file)).convert("RGB"),
                "is_document": "no",
            }
            count += 1
            if count >= MAX_PER_SOURCE:
                break

    count = 0
    # Processing documents
    for root, _, files in os.walk(RVL_CDIP):
        for file in files:
            if file.lower().endswith(".tif"):
                file_path = os.path.join(root, file)
                yield {
                    "image": Image.open(file_path).convert("RGB"),
                    "is_document": "yes",
                }
                count += 1
                if count >= MAX_PER_SOURCE:
                    break
            if count >= MAX_PER_SOURCE:
                break


features = Features(
    {"image": DatasetsImage(), "is_document": ClassLabel(names=["no", "yes"])}
)


def create_dataset():
    dataset = Dataset.from_generator(get_images, features=features)
    dataset = dataset.shuffle(seed=42)
    split_ratios = {"train": 0.8, "test": 0.1, "validation": 0.1}
    splits = dataset.train_test_split(
        test_size=split_ratios["test"] + split_ratios["validation"]
    )
    test_validation = splits["test"].train_test_split(
        test_size=split_ratios["validation"]
        / (split_ratios["test"] + split_ratios["validation"])
    )

    dataset_dict = DatasetDict(
        {
            "train": splits["train"],
            "test": test_validation["train"],
            "validation": test_validation["test"],
        }
    )
    return dataset_dict


if __name__ == "__main__":
    dataset_dict = create_dataset()
    dataset_dict.save_to_disk("./dataset")
    dataset_dict.push_to_hub("tarekziade/docornot")
