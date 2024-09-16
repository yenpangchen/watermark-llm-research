from datasets import load_dataset
import pandas as pd

# ds = load_dataset("allenai/real-toxicity-prompts")
# print(ds)


df = pd.read_csv('prompts/prompts_toxicity0.5.csv')
print(df['text'].head(1))

input_texts = df['text'].tolist()
print(input_texts)