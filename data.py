from datasets import load_dataset
import datasets
import torch

import numpy as np
from sklearn.metrics import accuracy_score

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score(predictions=predictions, references=labels)
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
        """
        Loads the Pokemon dataset and performs basic preprocessing.
        """
        self.data = load_dataset(self.data_dir, "full")

        # Combine train/validation/test sets if split is "train"

        self.data["train"] = datasets.concatenate_datasets([self.data["train"], self.data["validation"]])

        self.data["train"] = datasets.concatenate_datasets([self.data["train"], self.data["test"]])
        self.data = self.data['train'].train_test_split(test_size=self.test_size)
        self.data = self.data['train'].train_test_split(test_size=self.test_size)

        # Extract labels and create mappings
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


