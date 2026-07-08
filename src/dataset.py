import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    def __init__(self, text_ids, code_ids):
        self.text_ids = text_ids
        self.code_ids = code_ids
        assert len(text_ids) == len(code_ids)
    
    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, index):
        # convert numerical arrays to PyTorch tensors
        source_tensor = torch.tensor(self.text_ids[index], dtype=torch.long)
        target_tensor = torch.tensor(self.code_ids[index], dtype=torch.long)

        return source_tensor, target_tensor
    
def collate_fn(batch, pad_id=0):
    src, tgt = zip(*batch)

    # Calculate actual lengths of sequences before padding
    src_lengths = torch.tensor([len(s) for s in src], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt], dtype=torch.long)

    src_padded = pad_sequence(src, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt, batch_first=True, padding_value=pad_id)

    return src_padded, tgt_padded, src_lengths, tgt_lengths

def main():
    # Load the tokenized training and validation sets
    print("Loading tokenized training and validation sets...")
    with open('data/vocabulary/train_text_ids.pkl', 'rb') as f:
        train_text_ids = pickle.load(f)
    with open('data/vocabulary/train_code_ids.pkl', 'rb') as f:
        train_code_ids = pickle.load(f)
    with open('data/vocabulary/val_text_ids.pkl', 'rb') as f:
        val_text_ids = pickle.load(f)
    with open('data/vocabulary/val_code_ids.pkl', 'rb') as f:
        val_code_ids = pickle.load(f)
    

    train_dataset = CustomDataset(
        text_ids = train_text_ids,
        code_ids = train_code_ids,
    )

    val_dataset = CustomDataset(
        text_ids = val_text_ids,
        code_ids = val_code_ids
    )

    print(f'Total Train Dataset size: {len(train_dataset)}')
    print(f'Total Val Dataset size: {len(val_dataset)}')

     # Initialize PyTorch DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    # For Debugging
    print('\nVerifying Batch Tensor shapes')
    for src, tgt, src_lengths, tgt_lengths in train_loader:
        print(src.shape)
        print(tgt.shape)
        print(src_lengths)
        print(tgt_lengths)
        break

if __name__ == "__main__":
    main()
