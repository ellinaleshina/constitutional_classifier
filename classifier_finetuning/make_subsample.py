import pandas as pd

df = pd.read_csv("neutral_test.csv")

ss = df.sample(frac=0.1, random_state=42)

ss.to_csv('neutral_test_subsample.csv')