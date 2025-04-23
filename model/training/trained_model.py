from .config import GPT_SMALL 
from .gpt import GPTModel
import torch
import os
import urllib.request

SIMPLY_TRAINED_MODEL_CACHE_PATH = "simply-trained-model.pth"
SMALL_GPT_2_CACHE_PATH = "gpt2-small-124M.pth"
SMALL_GPT_2_REMOTE_PATH = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{SMALL_GPT_2_CACHE_PATH}"

def get_small_gpt_2_model():
    if not os.path.exists(SMALL_GPT_2_CACHE_PATH):
        urllib.request.urlretrieve(SMALL_GPT_2_REMOTE_PATH, SMALL_GPT_2_CACHE_PATH)
        print(f"Downloaded to {SMALL_GPT_2_CACHE_PATH}")
    
    model = GPTModel(GPT_SMALL)
    model.load_state_dict(torch.load(SMALL_GPT_2_CACHE_PATH, weights_only=True))
    model.eval()
    model.to(torch.device("cpu"))

    return model
