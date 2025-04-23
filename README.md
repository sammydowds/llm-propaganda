# LLM Fine-tuned to Classify Propaganda

## Summary

This is a simple POC / toy to fine-tune an LLM (GPT-2 in this case) for classifying propaganda. Right now it pulls the gpt2 weights and fine tunes that model with a limited sub set of data from a [weakely labeled propaganda dataset](https://github.com/leereak/propaganda-detection/blob/master/data/tweets/tweets.tsv).

## Training locally

Train and cache the model locally. Ensure you have `uv` installed.

```bash
cd model && uv sync && source .venv/bin/activate
```

```bash
python -m training.train 
```

This will output a file `propaganda-classfier.pth` with the pytorch state dict for future use.

It should take ~10 mins to train (MBP), sample training output:

```bash
Ep 2 (Step 000650): Train loss 0.159, Val loss 0.243
Ep 2 (Step 000700): Train loss 0.122, Val loss 0.333
Ep 2 (Step 000750): Train loss 0.050, Val loss 0.291
Ep 2 (Step 000800): Train loss 0.116, Val loss 0.253
Ep 2 (Step 000850): Train loss 0.039, Val loss 0.271
Training accuracy: 100.00% | Validation accuracy: 90.00%
Training completed: 9.05 minutes.
```

## Running the API

Ensure you have trained and saved the model with the steps above.

```bash
fastapi dev server/main.py
```

You should be able to visit [http://localhost:8000/docs](http://localhost:8000/docs#/default/classify_classify__post) and pass through payloads to test classification.

## Summary

Again, this is a toy project. When testing out classification you will probably find it useless. Training has not been tuned and this model + data is very small and inadiquate. Over time, I might:

1. Use a bigger model by default 
2. Train with GPU's in the cloud
3. Explore and improve accuracy testing/validation
4. Find better datasets, or create my own
5. Store the model in the cloud, so no training is necessary and you can just pull from there
6. Create a nice UI to interact

These would all improve this project quite a bit. 

