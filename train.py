import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CustomDataset, collate_fn
from models.attention import BahdanauAttention
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

import config

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for src, tgt, src_lenghts, tgt_lenghts in tqdm(dataloader, desc='Training', leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt, src_lenghts, teacher_forcing_ratio=0.5)
        output_dim = output.shape[-1]
        output = output[:,1:].reshape(-1, output_dim)
        tgt = tgt[:,1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for src, tgt, src_lenghts, tgt_lenghts in tqdm(dataloader, desc='Validation', leave=False):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, src_lenghts, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            tgt = tgt[:,1:].reshape(-1)
            loss = criterion(output, tgt)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

def plot_losses(train_losses, val_losses, output_image_path="checkpoints/loss_curve.png"):
    """Generates and saves visual tracking curves for auditing convergence trends."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image_path)
    plt.close()
    print(f"Loss curves plot accurately saved to {output_image_path}")


def main():
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Execution pipeline bound to active device: {device}')

    train_text_ids_path = os.path.join("data/vocabulary", "train_text_ids.pkl")
    train_code_ids_path = os.path.join("data/vocabulary", "train_code_ids.pkl")
    val_text_ids_path = os.path.join("data/vocabulary", "val_text_ids.pkl")
    val_code_ids_path = os.path.join("data/vocabulary", "val_code_ids.pkl")

    if not os.path.exists(train_text_ids_path) or not os.path.exists(train_code_ids_path):
        raise FileNotFoundError("Processed data not found. Please run preprocess.py first")
    else:
        with open(train_text_ids_path, 'rb') as f:
            train_text_ids = pickle.load(f)
        with open(train_code_ids_path, 'rb') as f:
            train_code_ids = pickle.load(f)

    if not os.path.exists(val_text_ids_path) or not os.path.exists(val_code_ids_path):
        raise FileNotFoundError("Processed data not found. Please run preprocess.py first")
    else:
        with open(val_text_ids_path, 'rb') as f:
            val_text_ids = pickle.load(f)
        with open(val_code_ids_path, 'rb') as f:
            val_code_ids = pickle.load(f)

    train_dataset = CustomDataset(train_text_ids, train_code_ids)
    val_dataset = CustomDataset(val_text_ids, val_code_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(config.text_vocab_size, config.embedding_dim, config.hidden_dim, config.pad_idx)
    attention = BahdanauAttention(config.hidden_dim)
    decoder = Decoder(config.code_vocab_size, config.embedding_dim, config.hidden_dim, attention, config.pad_idx)
    Model = Seq2Seq(encoder, decoder, device).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    optimizer = optim.Adam(Model.parameters(), lr=learning_rate)

    # Tracking arrays
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    save_model_path = 'checkpoints/best_model.pt'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2
    )

    print('\nInitiating Model Optimization Routine')
    for epoch in range(num_epochs):
        train_loss = train_epoch(Model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_epoch(Model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:02d}/{num_epochs}] "
            f"| LR: {current_lr:.6f} "
            f"| Train: {train_loss:.4f} "
            f"| Val: {val_loss:.4f}"
        )

        # save the best model
        if(val_loss < best_val_loss):
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                            "epoch": epoch,
                            "model_state_dict": Model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        },save_model_path,)
            
            print(f" => Validation loss decreased. Saving current weights to {save_model_path}")

        scheduler.step(val_loss)

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }

    with open("checkpoints/history.pkl", "wb") as f:
        pickle.dump(history, f)
    
    print("\nTraining completed safely.")
    # Plot final curves
    plot_losses(train_losses, val_losses) 
    print(f"\nBest validation loss : {best_val_loss:.4f}")
    print(f"Model saved to : {save_model_path}") 

if __name__=="__main__":
    main()


