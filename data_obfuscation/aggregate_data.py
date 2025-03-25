"""
Aggregates all the obfuscated data generated in obfuscated_data folder into one .csv file
"""

import os
import pandas as pd

data_path = "./obfuscated_data"
output_path = "./aggregated_data/aggregated.csv"

files = list(os.listdir(data_path))

df = pd.read_csv(os.path.join(data_path, files[0]))

for file in files:
    path = os.path.join(data_path, file)
    temp_df = pd.read_csv(path)
    df = pd.concat([df, temp_df], ignore_index=True)

df.to_csv(output_path, index=False)