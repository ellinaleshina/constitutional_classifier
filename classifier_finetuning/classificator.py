import torch
import torch.nn as nn

class Classificator:
    def __init__(self, model, tokenizer, prompt):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, prompt):
        return self.model.generate(prompt)

def CustomLoss(outputs, labels, response_token_dict):
    return nn.BCEWithLogitsLoss()(outputs, labels)