from pydantic import BaseModel
from fastapi import FastAPI
from .load_model import load_trained_classifier
from .inference import classify_text
import tiktoken
import torch

class Payload(BaseModel):
    text: str

device = torch.device("cpu")
tokenizer = tiktoken.get_encoding('gpt2')
model = load_trained_classifier()
app = FastAPI()

@app.post("/classify/")
async def classify(payload: Payload):
    text = payload.text
    result = classify_text(text, model, tokenizer, device)
    
    return result 
