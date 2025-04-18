import pandas as pd
import urllib
import zipfile
import os
from torch.utils.data import Dataset
import torch
from pathlib import Path

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
       
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        ) 
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


class PropagandaData():
    def __init__(self):
        self.url = "https://raw.githubusercontent.com/leereak/propaganda-detection/refs/heads/master/data/news/news-s.tsv"
        self.extracted_path = Path("news-s.tsv")

    def download(self):
        if self.extracted_path.exists():
            print(f"{self.extracted_path} already exists. Skipping download and extractions.")
            return

        with urllib.request.urlopen(self.url) as response:
            with open(self.extracted_path, "wb") as out_file:
                out_file.write(response.read())
        
        print(f"File downloaded and saved as {self.extracted_path}")

    def read(self):
        return pd.read_csv(
            self.extracted_path, sep="\t", header=None, names=["source", "label", "text"] 
        )

    def balance(self, df):
        num_propoganda = df[df["label"] == 1].shape[0]
        not_propoganda_subset = df[df["label"] == 0].sample(
            num_propoganda, random_state=123
        )
        balanced_df = pd.concat([
            not_propoganda_subset, df[df["label"] == 1]
        ])
        return balanced_df
    
    def random_split(self, df, train_frac = 0.7, validation_frac = 0.1):
        df = df.sample(
            frac=1, random_state=123
        ).reset_index(drop=True)

        train_end = int(len(df) * train_frac)
        validation_end = train_end + int(len(df) * validation_frac)
        train_df = df[:train_end]
        validation_df = df[train_end:validation_end]
        test_df = df[validation_end:]

        return train_df, validation_df, test_df
    
    def fetch_and_process(self):
        self.download()
        df = self.read()
        balanced_df = self.balance(df)

        train_df, validation_df, test_df = self.random_split(balanced_df, 0.7, 0.1)
        train_df.to_csv("train.csv", index=None)
        validation_df.to_csv("validation.csv", index=None)
        test_df.to_csv("test.csv", index=None)
