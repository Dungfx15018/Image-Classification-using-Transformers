# TODO 3
import os
import data
from argparse import ArgumentParser
from transformers import ViTImageProcessor,Trainer,ViTForImageClassification,TrainingArguments
import torch
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

from torchvision import transforms
os.environ["WANDB_DISABLED"] = "true"
if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--data-dir", default='fcakyon/pokemon-classification', type = str)
    parser.add_argument("--checkpoint", default='google/vit-base-patch16-224-in21k', type = str)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--gradient-accumulation-steps", default=4, type=int)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--per-device-train-batch-size", default=16, type=int)
    parser.add_argument("--per-device-eval-batch-size", default=16, type = int)
    parser.add_argument("--max-steps", default=2000, type=int)
    parser.add_argument("--num-train-epochs", default=20, type = int)
    parser.add_argument("--adam-epsilon", default=1e-8, type=float)
    parser.add_argument("--adam-beta1", default=0.9, type=float)
    parser.add_argument("--adam-beta2", default=0.99, type = float)
    parser.add_argument("--logging-steps", default=20, type = int)
    parser.add_argument("--warmup-ratio", default=0.1, type = float)
  

    home_dir = os.getcwd()
    args = parser.parse_args()

 
    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${account}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')

    data_dir = args.data_dir
    checkpoint = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = ViTImageProcessor.from_pretrained(checkpoint)


    datasets = data.DataProcessing(data_dir=data_dir, test_size= args.test_size)

    dataset = datasets.get_data()
    labels = datasets.get_labels()
    label2id = datasets.get_label2id()
    id2label = datasets.get_id2label()

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


    dataset = dataset.with_transform(transforms)
    #print(dataset['train'])


    model = ViTForImageClassification.from_pretrained(
        checkpoint,
        num_labels= len(labels),
        id2label = id2label,
        label2id = label2id
    ).to(device)


    training_args = TrainingArguments(
    output_dir="pokemon_models",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    max_steps=args.max_steps,
    num_train_epochs=args.num_train_epochs,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_epsilon=args.adam_epsilon,
    warmup_ratio=args.warmup_ratio,
    logging_steps= args.logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


    trainer = Trainer(
    model = model,
    data_collator=data.collate_fn,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset= dataset['test'],
    tokenizer=image_processor,
    compute_metrics=data.compute_metrics

)
    trainer.train()