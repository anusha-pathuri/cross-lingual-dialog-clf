import argparse
import os
import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import torch
import wandb
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, f1_score

torch.manual_seed(630)
np.random.seed(630)

gc.collect()
torch.cuda.empty_cache()

os.environ["WANDB_PROJECT"] = "SI630-Project"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using '{device}' device")   

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", required=True)
parser.add_argument("--train-filename", required=True)
parser.add_argument("--val-filename", required=True)
parser.add_argument("--test-filename", required=False)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--num-epochs", type=int, required=True)
parser.add_argument("--eval-steps", type=int, required=False, default=100)
parser.add_argument("--save-steps", type=int, required=False, default=100)
parser.add_argument("--model-name", required=True, choices=["roberta", "camembert", "xlm-roberta"])
parser.add_argument("--language", required=True, choices=["english", "french"])
parser.add_argument("--model-max-length", type=int, required=False, default=50)
parser.add_argument("--train-batch-size", type=int, required=False, default=32)
parser.add_argument("--eval-batch-size", type=int, required=False, default=64)
parser.add_argument("--learning-rate", type=float, required=False, default=2e-5)

args = parser.parse_args()

HF_MODELS = {
    "roberta": "FacebookAI/roberta-base",
    "camembert": "almanach/camembert-base",
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
}
model_name = HF_MODELS.get(args.model_name)
if model_name is None:
    raise NotImplementedError(args.model_name)


# Load dataset
train_df = pd.read_csv(os.path.join(args.dataset_dir, args.train_filename))
val_df = pd.read_csv(os.path.join(args.dataset_dir, args.val_filename))
if args.test_filename is not None:
    test_df = pd.read_csv(os.path.join(args.dataset_dir, args.test_filename))

id2label = train_df[["label", "label_text"]].drop_duplicates().set_index("label")["label_text"].to_dict()
id2label = dict(sorted(id2label.items(), key=lambda item: item[0]))
label2id = {l: i for i, l in id2label.items()}
class_names = list(label2id.keys())
print(f"{len(class_names)} classes:", class_names)

if args.language == "english":
    text_col = "text_en"
elif args.language == "french":
    text_col = "text_fr"
else:
    raise NotImplementedError(args.language)

cols = ["id", text_col, "label"]
train_ds = Dataset.from_pandas(train_df[cols]).rename_column("label", "labels")
val_ds = Dataset.from_pandas(val_df[cols]).rename_column("label", "labels")
if args.test_filename is not None:
    test_ds = Dataset.from_pandas(test_df[cols]).rename_column("label", "labels")


# Preprocess data
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=args.model_max_length)
print(f"Initializer tokenizer for {model_name}")

def tokenize(examples, tokenizer=tokenizer, text_col=text_col):
    tokens = tokenizer(examples[text_col], padding="longest", return_tensors="pt")
    examples.update(tokens)
    return examples

train_ds = train_ds.map(tokenize, batched=True, batch_size=len(train_ds))
val_ds = val_ds.map(tokenize, batched=True, batch_size=len(val_ds))
if args.test_filename is not None:
    test_ds = test_ds.map(tokenize, batched=True, batch_size=len(test_ds))


# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
)
print(f"Loaded model from {model_name}")
print(model.config)

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

print_trainable_parameters(model)


# Training
os.makedirs(args.output_dir, exist_ok=True)
run_dir = os.path.join(args.output_dir, args.run_name)
print(f"Saving outputs to {run_dir}")

training_args = TrainingArguments(
    output_dir=run_dir,
    overwrite_output_dir=True,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    do_eval=True,
    seed=630,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=2,
    num_train_epochs=args.num_epochs,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    report_to="wandb",
    run_name=args.run_name,
)

def compute_metrics(eval_pred):
    true = eval_pred.label_ids.astype(int)
    preds = eval_pred.predictions.argmax(-1)
    precision, recall, micro_f1, _ = precision_recall_fscore_support(true, preds, average="micro")
    macro_f1 = f1_score(true, preds, average="macro")
    weighted_f1 = f1_score(true, preds, average="weighted")
    return {"f1": micro_f1, "precision": precision, "recall": recall, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


# Fin
wandb.finish()
gc.collect()
torch.cuda.empty_cache()
