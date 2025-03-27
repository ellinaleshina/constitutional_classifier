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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Available device:", device)

# Model configuration
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESTRICTED_CLASS = "Tarantino"
OUTPUT_DIR = "./tarantino-classifier"
DATA_PATH = "train_subsample.csv"
PROMPT_PATH = "input_classifier_prompt.txt"

# Training configuration
TRAIN_BATCH_SIZE = 3
EVAL_BATCH_SIZE = 3
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
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

# Prepare dataloader collator
def collate_fn(batch, tokenizer):
    inputs = [item["formatted_prompt"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    
    return {"labels": labels.to(device), "inputs": tokenized_inputs.to(device)}

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,             # Rank of LoRA adapters
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
train_data, test_data = train_test_split_csv(DATA_PATH, train_ratio=0.85, seed=42)
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

# Setup optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_training_steps = NUM_EPOCHS * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=num_training_steps
)

# Initialize wandb
wandb.init(project="tarantino-classifier")

# Training loop
best_accuracy = 0.0

# Setup scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler() if FP16 else None

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for batch in progress_bar:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        
        # Get batch data
        labels = batch["labels"]
        inputs = batch["inputs"]
        
        # Forward pass with mixed precision if available
        if FP16:
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                # Get logits for the last token position only
                logits = outputs.logits[:, -1, :]
                # Cross entropy loss
                loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            raise ValueError("FP16 = CUDA is not enabled")
        
        lr_scheduler.step()
        train_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
    
    # Calculate average training loss for the epoch
    avg_train_loss = train_loss / len(train_dataloader)
    
    # Evaluation
    model.eval()
    eval_loss = 0.0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Eval]")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Get batch data
            labels = batch["labels"]
            inputs = batch["inputs"]
            
            # Forward pass
            outputs = model(**inputs)
            
            # Get logits for the last token position only
            logits = outputs.logits[:, -1, :]
            
            # Cross entropy loss
            loss = torch.nn.functional.cross_entropy(logits, labels)
            eval_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Store predictions and labels for metric computation
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average evaluation loss
    avg_eval_loss = eval_loss / len(eval_dataloader)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels)
    accuracy_score = metrics["accuracy"]
    
    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "eval_loss": avg_eval_loss,
        "accuracy": accuracy_score
    })
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Eval Loss: {avg_eval_loss:.4f}")
    print(f"  Accuracy: {accuracy_score:.4f}")
    
    # Save the best model
    if accuracy_score > best_accuracy:
        best_accuracy = accuracy_score
        print(f"  New best accuracy: {best_accuracy:.4f}! Saving model...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

# Close wandb
wandb.finish()