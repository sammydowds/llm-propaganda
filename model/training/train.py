from .loss import calc_accuracy_loader, calc_loss_batch, evaluate_model
from .plot import plot_values
from .trained_model import get_small_gpt_2_model
import torch
from .config import GPT_SMALL
import tiktoken
from torch.utils.data import DataLoader 
import time
from .datasets import PropagandaDataset, PropagandaData 

TRAINED_MODEL_CACHE = 'propaganda-classifier.pth'

def train(
        model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
    ):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def train_classifier(skip_cache = False):
    # init model 
    gpt = get_small_gpt_2_model()
    torch.manual_seed(123)
    
    # tweak for classification
    num_classes = 2
    gpt.out_head = torch.nn.Linear(
        in_features=GPT_SMALL["emb_dim"],
        out_features=num_classes,
    )
    for param in gpt.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in gpt.final_norm.parameters():
        param.requires_grad = True

    # check for trained cache
    cached = None
    if not skip_cache:
        try: 
            cached = torch.load(TRAINED_MODEL_CACHE)
        except:
            print("Unabled to find cached fine-tuned classifier model.")
        if cached:
            print("Found cached model, skipping classification fine tuning.")
            gpt.load_state_dict(cached)
            gpt.eval()
            return gpt
    
    # no cache, train 
    fetcher = PropagandaData()
    fetcher.fetch_and_process()
   
    tokenizer = tiktoken.get_encoding('gpt2')
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 2 
    device = torch.device("cpu")

    train_dataset = PropagandaDataset(
        csv_file='train.csv',
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = PropagandaDataset(
        csv_file='validation.csv',
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=8,
        num_workers=0,
        drop_last=False
    )
    
    train_losses, val_losses, train_accs, val_accs, examples_seen = train(
        gpt, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    # plot training results
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="Accuracy")

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed: {execution_time_minutes:.2f} minutes.")

    # cache trained model
    if not skip_cache:
        torch.save(gpt.state_dict(), TRAINED_MODEL_CACHE)
        print(f"Trained model cached to {TRAINED_MODEL_CACHE}")
     
if __name__ == "__main__":
    train_classifier()
