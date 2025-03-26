import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_scheduler,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
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

os.environ["WANDB_PROJECT"] = "tarantino-classifier"
CUDA_DEVICE = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
device = torch.device("cuda:" + CUDA_DEVICE if torch.cuda.is_available() else "cpu")
print("Available device:", device)


# Model configuration
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESTRICTED_CLASS = "Tarantino"
OUTPUT_DIR = "./tarantino-classifier"
DATA_PATH = "aggregated.csv"
PROMPT_PATH = "input_classifier_prompt.txt"

# Training configuration
TRAIN_BATCH_SIZE = 3
EVAL_BATCH_SIZE = 3
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0
FP16 = torch.cuda.is_available()

# Load classifier prompt
classifier_prompt = open(PROMPT_PATH, "r").read()

# Initialize accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(predictions, labels):
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    }

# Modify the collate_fn to work with Trainer
def collate_fn(batch, tokenizer):
    inputs = [item["formatted_prompt"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    
    # Convert labels to tensor
    # Convert labels to one-hot encoding
    batch_size = len(labels)
    vocab_size = tokenizer.vocab_size
    one_hot_labels = torch.zeros(batch_size, vocab_size, dtype=torch.float)
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1.0
    labels = one_hot_labels
    
    return {"labels": labels, **tokenized_inputs}

# Compute metrics function for the Trainer
def compute_metrics_for_trainer(eval_pred):
    logits, labels = eval_pred
    # Get predictions from the last token position
    predictions = logits[:, -1, :].argmax(axis=-1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"]}

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,             # Rank of LoRA adapters
    lora_alpha=32,    # LoRA scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

# Create LoRA model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(device)

# Load and prepare data
train_data, test_data = train_test_split_csv(DATA_PATH, train_ratio=0.8, seed=42)
train_dataset = TrainObfuscatedPromptDataset(train_data, tokenizer, classifier_prompt)
test_dataset = TestObfuscatedPromptDataset(test_data, tokenizer, classifier_prompt)

# Create dataloaders
train_collate_fn = partial(collate_fn, tokenizer=tokenizer)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=TRAIN_BATCH_SIZE, 
    shuffle=True, 
    collate_fn=train_collate_fn,
    num_workers=0
)

eval_dataloader = DataLoader(
    test_dataset, 
    batch_size=EVAL_BATCH_SIZE, 
    collate_fn=train_collate_fn,
    num_workers=0
)

# Initialize wandb
wandb.init(project="tarantino-classifier")

# Set up training arguments
training_args = TrainingArguments(
    
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    dataloader_num_workers=0,
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=FP16,
    report_to="wandb",
    learning_rate=LEARNING_RATE,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    
    tokenizer=tokenizer,
    data_collator=lambda batch: collate_fn(batch, tokenizer),
    compute_metrics=compute_metrics_for_trainer,
)

# Train the model
trainer.train()

# Save the best model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Close wandb
wandb.finish()

# Example of using the finetuned model for prediction
def predict(text):
    formatted_text = f"Prompt: {text} Answer:"
    inputs = tokenizer(formatted_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1,
            temperature=0
        )
    predicted_token_id = outputs[0][-1].item()
    is_tarantino = predicted_token_id == tokenizer.convert_tokens_to_ids(" yes")
    return is_tarantino, "Tarantino" if is_tarantino else "Not Tarantino"

# Example predictions
test_texts = [
    "Quentin Tarantino directed Pulp Fiction and Django Unchained.",
    "Christopher Nolan is known for Inception and The Dark Knight trilogy."
]

for text in test_texts:
    result, label = predict(text)
    print(f"Text: {text}\nClassification: {label}\n")
