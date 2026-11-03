import ast, json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from .config import MAX_POSTS, MAX_TOKENS, SEED

def parse_list(value):
    if isinstance(value, list): return value
    if pd.isna(value) or not str(value).strip(): return []
    try: result = json.loads(value)
    except (json.JSONDecodeError, TypeError): result = ast.literal_eval(str(value))
    return result if isinstance(result, list) else [str(result)]

def load_data(path):
    df = pd.read_csv(path)
    if not {"label", "posts"}.issubset(df.columns):
        raise ValueError("CSV must contain label and posts columns.")
    df["label"] = df["label"].astype(int)
    if "timestamps" in df:
        post_lists = df["posts"].apply(parse_list)
        time_lists = df["timestamps"].apply(parse_list)
        selected = []
        for posts, times in zip(post_lists, time_lists):
            pairs = list(zip(times, posts)) if len(times) == len(posts) else []
            if pairs: posts = [p for _, p in sorted(pairs, key=lambda x: str(x[0]))]
            selected.append([str(p) for p in posts if str(p).strip()][-MAX_POSTS:])
        df["selected_posts"] = selected
    else:
        df["selected_posts"] = df["posts"].apply(
            lambda x: [str(p) for p in parse_list(x) if str(p).strip()][-MAX_POSTS:]
        )
    return df[df.selected_posts.map(len) > 0].reset_index(drop=True)

def split_data(df):
    train, temp = train_test_split(df, test_size=.30, stratify=df.label, random_state=SEED)
    val, test = train_test_split(temp, test_size=.50, stratify=temp.label, random_state=SEED)
    return [x.reset_index(drop=True) for x in (train, val, test)]

class UserDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True); self.tokenizer = tokenizer
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        posts = row.selected_posts[-MAX_POSTS:]
        n = len(posts)
        posts += [""] * (MAX_POSTS - n)
        enc = self.tokenizer(posts, padding="max_length", truncation=True,
                             max_length=MAX_TOKENS, return_tensors="pt")
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask,
                "post_mask": torch.tensor([1.]*n + [0.]*(MAX_POSTS-n)),
                "label": torch.tensor(int(row.label), dtype=torch.long)}
