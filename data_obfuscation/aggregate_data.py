"""
Aggregates all the obfuscated data generated in obfuscated_data folder into one .csv file
"""

import os
import pandas as pd

data_path = "/workspace/mnt/local/data/sharov/constitutional_classifier/synthetic_prompts_via_imdb/generation/corrupted_data/"
output_path = "/workspace/mnt/local/data/sharov/constitutional_classifier/synthetic_prompts_via_imdb/generation/corrupted_data/aggregated_train.csv"


test_files = ['Anderson_corrupted.csv', 'Tarantino_test.csv', 'neutral_test.csv']

restricted_files = ['aggregated_train.csv', 'aggregated_test.csv']

files = list(os.listdir(data_path))

df = pd.read_csv(os.path.join(data_path, files[0]))

for file in files:
    if file in restricted_files:
        continue
    if file in test_files:
        continue
    print(file)
    path = os.path.join(data_path, file)
    temp_df = pd.read_csv(path)
    df = pd.concat([df, temp_df], ignore_index=True)

df.to_csv(output_path, index=False)