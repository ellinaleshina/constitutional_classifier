import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,

)
from peft import get_peft_model, LoraConfig, TaskType
from dataset import TrainObfuscatedPromptDataset, TestObfuscatedPromptDataset, train_test_split_csv
import evaluate
from functools import partial
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader
import wandb
import os
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def collate_fn(batch, tokenizer):
    inputs = [item["formatted_prompt"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    
    return {"labels": labels.to(device), "inputs": tokenized_inputs.to(device)}


accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
def compute_metrics(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    return {
        # Assuming being flagged as tarantino is the positive class
        # Averaging is done globally over the whole dataset
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "FPR": 1-precision.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "TPR": recall.compute(predictions=predictions, references=labels, average="macro")["recall"]
    }


os.environ["WANDB_PROJECT"] = "tarantino-classifier-test"


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
DATA_PATH = "./aggregated_test.csv"
RUN_NAME = "aggregated_macro"

PROMPT_PATH = "input_classifier_prompt.txt"
EVAL_BATCH_SIZE = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("./tarantino-classifier")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(base_model, "./tarantino-classifier/")

# model = AutoModelForCausalLM.from_pretrained(MODEL)

model.to(device)
model.eval()  # Set to evaluation mode


classifier_prompt = open(PROMPT_PATH, "r").read()
test_data = pd.read_csv(DATA_PATH)
test_dataset = TestObfuscatedPromptDataset(test_data, tokenizer, classifier_prompt)

train_collate_fn = partial(collate_fn, tokenizer=tokenizer)

eval_dataloader = DataLoader(
    test_dataset, 
    batch_size=EVAL_BATCH_SIZE, 
    collate_fn=train_collate_fn,
    num_workers=0
)


wandb.init(project="tarantino-classifier-test", name=RUN_NAME)


    # Train
all_predictions = []
all_labels = []

progress_bar = tqdm(eval_dataloader)

with torch.no_grad():
    for batch in progress_bar:
        # Get batch data
        labels = batch["labels"]
        inputs = batch["inputs"]
        
        # Forward pass
        outputs = model(**inputs)
        
        # Get logits for the last token position only
        logits = outputs.logits[:, -1, :]
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1) 
        
        # Store predictions and labels for metric computation
        all_predictions.extend(predictions.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())

        # print(tokenizer.convert_ids_to_tokens(predictions.cpu().numpy()))
        # print(labels.cpu().numpy())



# Compute metrics
metrics = compute_metrics(all_predictions, all_labels)
accuracy_score = metrics["accuracy"]
fpr = metrics["FPR"]
tpr = metrics["TPR"]

# Log to wandb
wandb.log({
    "accuracy": accuracy_score,
    "FPR": fpr,
    "TPR": tpr
})

print(f"  Accuracy: {accuracy_score:.4f}")
wandb.finish()