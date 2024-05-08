from datasets import load_dataset
import datasets
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import ViTImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])


def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))
class DataProcessing():
    def __init__(self, data_dir, test_size=0.4):

        self.data_dir = data_dir
        self.test_size = test_size
        self.data = None
        self.labels = None
        self.label2id = None
        self.id2label = None
        self.load_data()

    def load_data(self):

        self.data = load_dataset(self.data_dir, "full")



        self.data["train"] = datasets.concatenate_datasets([self.data["train"], self.data["validation"]])

        self.data["train"] = datasets.concatenate_datasets([self.data["train"], self.data["test"]])
        self.data = self.data['train'].train_test_split(test_size=self.test_size).shuffle()
        self.data = self.data['train'].train_test_split(test_size=self.test_size).shuffle()


        self.labels = self.data['train'].features["labels"].names
        self.label2id = {label: str(i) for i, label in enumerate(self.labels)}
        self.id2label = {str(i): label for i, label in enumerate(self.labels)}

    def get_data(self):

        return self.data

    def get_labels(self):

        return self.labels

    def get_label2id(self):

        return self.label2id

    def get_id2label(self):
        return self.id2label


