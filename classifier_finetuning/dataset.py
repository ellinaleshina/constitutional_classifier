import torch
from torch.utils.data import Dataset
import random
import pandas as pd
def train_test_split_csv(data_path, train_ratio=0.8, seed=42):
    data = pd.read_csv(data_path)
    train_data = data.sample(frac=train_ratio, random_state=seed)
    test_data = data.drop(train_data.index)
    return train_data, test_data


class TestObfuscatedPromptDataset(Dataset):
    def __init__(self, data, tokenizer, classifier_prompt, restricted_class="Tarantino", yes_word="Yes", no_word="No"):
        self.tokenizer = tokenizer
        self.classifier_prompt = classifier_prompt
        self.yes_token_id = tokenizer.convert_tokens_to_ids(yes_word)
        self.no_token_id = tokenizer.convert_tokens_to_ids(no_word)
        self.dataset = []
        
        # Process existing data
        for _, row in data.iterrows():
            self.dataset.append((row['obfuscated_prompt'], 
                                self.yes_token_id if row['class'] == restricted_class else self.no_token_id,
                                row['original_prompt'], 
                                row['class'], 
                                row['primitiv_index']))
    
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        obfuscated_prompt, label, *_ = self.dataset[idx]
        formatted_prompt = self.classifier_prompt.replace("{prompt}", str(obfuscated_prompt))
        return {
            "formatted_prompt": formatted_prompt,
            "label": label
        }
    

class TrainObfuscatedPromptDataset(Dataset):
    def __init__(self, data, tokenizer, classifier_prompt, restricted_class="Tarantino", yes_word="Yes", no_word="No"):
        self.tokenizer = tokenizer
        self.classifier_prompt = classifier_prompt
        self.yes_token_id = tokenizer.convert_tokens_to_ids(yes_word)
        self.no_token_id = tokenizer.convert_tokens_to_ids(no_word)
        unique_prompts = data['original_prompt'].unique()
        self.dataset = {prompt: [] for prompt in unique_prompts}
        for _, row in data.iterrows():
            self.dataset[row['original_prompt']].append((row['obfuscated_prompt'], self.yes_token_id if row['class'] == restricted_class else self.no_token_id, row['original_prompt'], row['class'], row['primitiv_index']))
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        prompt = list(self.dataset.keys())[idx]
        obfuscated_prompt, label, *_ = random.choice(self.dataset[prompt])
        formatted_prompt = self.classifier_prompt.replace("{prompt}", str(obfuscated_prompt))
        return {
            "formatted_prompt": formatted_prompt,
            "label": label
        }
    
    