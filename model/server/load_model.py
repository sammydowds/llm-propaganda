from training.config import GPT_SMALL 
from training.gpt import GPTModel
from training.train import TRAINED_MODEL_CACHE
import torch
import os

RELATIVE_PATH_MODEL_TRAINING = "../training/"

def load_trained_classifier():
    if not os.path.exists(RELATIVE_PATH_MODEL_TRAINING + TRAINED_MODEL_CACHE):
        print(f"Unable to find trained model...")
        return
    
    model = GPTModel(GPT_SMALL)
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=GPT_SMALL["emb_dim"],
        out_features=num_classes,
    )
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    model.load_state_dict(torch.load(TRAINED_MODEL_CACHE, weights_only=True))
    model.eval()
    model.to(torch.device("cpu"))

    return model
