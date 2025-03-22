import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, TaskType

class CustomLoss(nn.Module):
    def __init__(self, lambda_val=0.1):
        super().__init__()
        self.lambda_val = lambda_val
    
    def forward(self, outputs, labels):
        # Next token prediction loss
        ntp_loss = nn.CrossEntropyLoss()(outputs.logits[:, :-1], labels[:, 1:])
        
        # Binary cross entropy for classification
        logits = outputs.logits[:, -1]  # Use last token for classification
        bce_loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        
        return self.lambda_val * ntp_loss + bce_loss

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("QwenLM/qwen-350m")
model = AutoModelForCausalLM.from_pretrained("QwenLM/qwen-350m")

# Configure LoRA
lora_config = PeftConfig(
    task_type=TaskType.CAUSAL_LM,
    adapter_name="lora",
    r=16,  # Rank of LoRA adapters
    target_modules=[nn.Linear],
    lora_alpha=32
)

# Create LoRA model
model_lora = PeftModel.from_pretrained(
    model=model,
    config=lora_config,
    device_map="auto",  # Automatic GPU memory management
    torch_dtype=torch.float16  # Use mixed precision
)

# Set up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lora.to(device)
optimizer = torch.optim.Adam(model_lora.parameters(), lr=1e-4)
loss_fn = CustomLoss(lambda_val=0.1)

# Training loop
for batch in train_dataloader:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    optimizer.zero_grad()
    
    outputs = model_lora(input_ids=input_ids, labels=input_ids)
    loss = loss_fn(outputs, labels)
    
    loss.backward()
    optimizer.step()
