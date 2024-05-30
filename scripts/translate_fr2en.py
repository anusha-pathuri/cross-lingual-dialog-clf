"""Machine Translation"""
import argparse
import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import NllbTokenizer
from transformers import AutoModelForSeq2SeqLM

gc.collect()
torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument("--input-path", required=True)
parser.add_argument("--output-path", required=True)
parser.add_argument("--output-column", required=False, default="text_fr2en")
parser.add_argument("--model-name", required=False, default="nllb")

args = parser.parse_args()


def create_dataset(df, cols=["id", "text_fr", "label"]):
    dataset = Dataset.from_pandas(df[cols])
    if "label" in cols:
        dataset = dataset.rename_column("label", "labels")
    return dataset


def preprocess(examples, tokenizer, text_col="text_fr"):
    inputs = tokenizer(examples[text_col], padding="longest", return_tensors="pt")
    examples.update(inputs)
    return examples


def translate(texts, model, tokenizer, text_col="text_fr", tgt_lang="eng_Latn", batch_size=32, max_length=40):
    if isinstance(texts, pd.DataFrame):
        dataset = create_dataset(texts, cols=[text_col])
    elif isinstance(texts, list):
        dataset = Dataset.from_dict({text_col: texts})
    else:
        raise NotImplementedError

    preprocessor = partial(preprocess, tokenizer=tokenizer, text_col=text_col)
    dataset = dataset.map(preprocessor, batched=True)
    dataset = dataset.remove_columns(text_col)
    collate_fn = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    outputs = []
    for batch in dataloader:
        translated_tokens = model.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], max_length=max_length
        )
        outputs.extend(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True))

    return outputs


if args.model_name == "nllb":
    model_name = "facebook/nllb-200-distilled-600M"
    Tokenizer = NllbTokenizer
else:
    raise NotImplementedError

tokenizer_fr = Tokenizer.from_pretrained(model_name, src_lang="fra_Latn", tgt_lang="eng_Latn")
model_fr = AutoModelForSeq2SeqLM.from_pretrained(model_name)

try:
    df = pd.read_csv(args.input_path)
except Exception as err:
    print(err)

df[args.output_column] = translate(df, model_fr, tokenizer_fr, batch_size=64)
df.to_csv(args.output_path, index=False)
print(f"Saved translations to {args.output_path}")

del model_fr, tokenizer_fr
gc.collect()
torch.cuda.empty_cache()
