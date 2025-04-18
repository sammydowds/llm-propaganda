from trained_model import get_small_gpt_2_model
import torch
from configs import GPT_SMALL
import tiktoken
from torch.utils.data import DataLoader 
import time
from fine_tuning.classification.dataset import SpamDataset
from fine_tuning.classification.fetcher import SMSSpamFetcher
import matplotlib.pyplot as plt

TRAINED_MODEL_CACHE = 'propaganda-classifier.pth'

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions / num_examples


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(
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
                    f"Val loss {train_loss:.3f}"
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

def get_trained_spam_classifier(skip_cache = False):
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
    fetcher = SMSSpamFetcher()
    fetcher.fetch_and_process()
   
    tokenizer = tiktoken.get_encoding('gpt2')
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    device = torch.device("cpu")

    train_dataset = SpamDataset(
        csv_file='train.csv',
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file='validation.csv',
        max_length=None,
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
    
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
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
     
    return gpt

def classify_text(
    text, model, tokenizer, device, max_length=None, pad_token_id=50256
):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    if max_length:
        input_ids = input_ids[:min(
            max_length, supported_context_length
        )]
        input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


def plot_values(
    epochs_seen, examples_seen, train_values, val_values, label="loss"
):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, label=f"Validation {label}", linestyle="-.")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
