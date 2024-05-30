import argparse
import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

torch.manual_seed(630)
np.random.seed(630)

gc.collect()
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using '{device}' device")   

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--text-column", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--output-path", required=True)

args = parser.parse_args()

df = pd.read_csv(args.dataset_path)

model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=device)

preds = classifier(df[args.text_column].tolist(), return_all_scores=False, function_to_apply="softmax")
preds_df = pd.concat([df[["id"]], pd.DataFrame(preds)], axis=1)
preds_df["label"] = preds_df["label"].map(model.config.label2id)

preds_df.to_csv(args.output_path, index=False)
print(f"Saved predictions to {args.output_path}")

del model
gc.collect()
torch.cuda.empty_cache()
