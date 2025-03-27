import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, )

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROMPT_PATH = "input_classifier_prompt.txt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("./tarantino-classifier-balanced")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
# Load LoRA adapters into the base model
model = PeftModel.from_pretrained(base_model, "./tarantino-classifier-balanced")

model.to(device)
model.eval()  # Set to evaluation mode

classifier_prompt = open(PROMPT_PATH, "r").read()

finish = False if args.i else True

while True:
    prompt = input("\nPlease, input your query:\n")

    if prompt == 'end':
        break

    formatted_prompt = classifier_prompt.replace("{prompt}", str(prompt))
    tokenized_input = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    outputs = model(**tokenized_input)
    logits = outputs.logits[:, -1, :]
    predictions = torch.argmax(logits, dim=-1)

    answer = tokenizer.convert_ids_to_tokens(predictions.cpu().numpy())

    if answer[0] == "Yes":
        print("Tarantino")

    elif answer[0] == "No":
        print("Not Tarantino")

    else:
        print("Model's broken, sorry...")

    if finish:
        break


