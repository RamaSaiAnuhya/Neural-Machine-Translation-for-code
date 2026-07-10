import os
import pickle
import random
import ast
import config
from itertools import zip_longest
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.dataset import CustomDataset, collate_fn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from src.models.attention import BahdanauAttention
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq

checkpoint_path = 'checkpoints/best_model.pt'
output_dir = 'results'

os.makedirs(output_dir, exist_ok = True)

def greedy_decode(model, src, src_lengths, max_len, sos_idx, eos_idx, device):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src, src_lengths)
        input_token = torch.tensor([sos_idx] * src.size(0), device=device)

        predictions = []
        for _ in range(max_len):
            prediction, hidden, cell, _ = model.decoder(input_token, (hidden, cell), encoder_outputs)
            input_token = prediction.argmax(1)
            predictions.append(input_token.unsqueeze(1))

            # Stop if all sequences predicted <EOS>
            if (input_token == eos_idx).all():
                break

        return torch.cat(predictions, dim=1)


def beam_search_decode(model, src, src_lengths, max_len, sos_idx, eos_idx, pad_idx, device, beam_size=5):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src, src_lengths)
        mask = (src != pad_idx).int()

        # Initialize beam with SOS
        beams = [(torch.tensor([sos_idx], device=device), hidden, cell, 0.0)]  # (sequence, hidden, cell, score)

        for _ in range(max_len):
            new_beams = []
            for seq, h, c, score in beams:
                input_token = seq[-1].unsqueeze(0)  # last token
                if input_token.item() == eos_idx:
                    new_beams.append((seq, h, c, score))
                    continue

                prediction, h_new, c_new, _ = model.decoder(input_token, (h, c), encoder_outputs, mask)
                log_probs = torch.log_softmax(prediction, dim=1).squeeze(0)

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                for k in range(beam_size):
                    new_seq = torch.cat([seq, topk_indices[k].unsqueeze(0)])
                    new_score = score + topk_log_probs[k].item()
                    new_beams.append((new_seq, h_new, c_new, new_score))

            # Keep top beam_size sequences
            beams = sorted(new_beams, key=lambda x: x[3], reverse=True)[:beam_size]

        # Return best sequence
        best_seq = beams[0][0]
        return best_seq.cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    text_word2idx_path = os.path.join("data/vocabulary", "text_word2idx.pkl")
    text_idx2word_path = os.path.join("data/vocabulary", "text_idx2word.pkl")
    code_word2idx_path = os.path.join("data/vocabulary", "code_word2idx.pkl")
    code_idx2word_path = os.path.join("data/vocabulary", "code_idx2word.pkl")

    val_text_ids_path = os.path.join("data/vocabulary", "val_text_ids.pkl")
    val_code_ids_path = os.path.join("data/vocabulary", "val_code_ids.pkl")

    if not os.path.exists(text_word2idx_path) or not os.path.exists(text_idx2word_path):
        raise FileNotFoundError("Processed data not found. Please run preprocess.py first")
    else:
        with open(text_word2idx_path, 'rb') as f:
            text_word2idx = pickle.load(f)
        with open(text_idx2word_path, 'rb') as f:
            text_idx2word = pickle.load(f)

    if not os.path.exists(code_word2idx_path) or not os.path.exists(code_idx2word_path):
        raise FileNotFoundError("Processed data not found. Please run preprocess.py first")
    else:
        with open(code_word2idx_path, 'rb') as f:
            code_word2idx = pickle.load(f)
        with open(code_idx2word_path, 'rb') as f:
            code_idx2word = pickle.load(f)

    print(f"Text vocabulary : {len(text_word2idx)}")
    print(f"Code vocabulary : {len(code_word2idx)}")

    if not os.path.exists(val_text_ids_path) or not os.path.exists(val_code_ids_path):
        raise FileNotFoundError("Processed data not found. Please run preprocess.py first")
    else:
        with open(val_text_ids_path, 'rb') as f:
            val_text_ids = pickle.load(f)
        with open(val_code_ids_path, 'rb') as f:
            val_code_ids = pickle.load(f)

    val_dataset = CustomDataset(val_text_ids, val_code_ids)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    print("Building model...")

    encoder = Encoder(
    text_vocab_size=config.text_vocab_size,
    embedding_dim=config.embedding_dim,
    hidden_dim=config.hidden_dim,
    pad_idx=config.pad_idx,
    num_layers=config.num_layers,
    dropout=config.dropout,
    )

    attention = BahdanauAttention(config.hidden_dim * 2, config.hidden_dim)
    
    decoder = Decoder(
    code_vocab_size=config.code_vocab_size,
    embedding_dim=config.embedding_dim,
    hidden_dim=config.hidden_dim,
    attention=attention,
    pad_idx=config.pad_idx,
    num_layers=config.num_layers,
    dropout=config.dropout,
    )

    model = Seq2Seq(
    encoder,
    decoder,
    device,
    ).to(device)

    print("Loading checkpoint...")

    checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Choose decoding method
    use_beam_search = True  # set False to use greedy
    print("Running inference with beam search..." if use_beam_search else "Running inference with greedy decoding...")

    print("Running inference (Teacher Forcing = 0)...")
    
    raw_intents = []
    ground_truths = []
    predictions = []

    # Helper helper to convert list of IDs to list of tokens
    SPECIAL_TOKENS = {
    "<SOS>",
    "<EOS>",
    "<PAD>",
    }


    def ids_to_tokens(ids, idx2word):
        tokens = []

        for idx in ids:

            token = idx2word.get(int(idx), "<UNK>")

            if token == "<EOS>":
                break

            if token in SPECIAL_TOKENS:
                continue

            tokens.append(token)

        return tokens

    with torch.no_grad():
        for src, trg, src_lengths, tgt_lengths in tqdm(val_loader, desc='Evaluating'):
            src = src.to(device)
            trg = trg.to(device)

            if use_beam_search:
                preds = beam_search_decode(
                    model, src, src_lengths, max_len=trg.size(1),
                    sos_idx=code_word2idx["<SOS>"], eos_idx=code_word2idx["<EOS>"],
                    pad_idx=config.pad_idx, device=device, beam_size=4
                )
                preds = [preds]  # single sequence per batch
            else:
                preds = greedy_decode(
                    model, src, src_lengths, max_len=trg.size(1),
                    sos_idx=code_word2idx["<SOS>"], eos_idx=code_word2idx["<EOS>"],
                    pad_idx=config.pad_idx, device=device
                ).cpu().numpy()

            src_ids = src.cpu().numpy()
            trg_ids = trg.cpu().numpy()

            for i in range(len(src_ids)):
                # Convert source sequence
                intent_tokens = ids_to_tokens(src_ids[i], text_idx2word)
                raw_intents.append(" ".join(intent_tokens))

                # Convert target/ground truth sequence
                gt_tokens = ids_to_tokens(trg_ids[i], code_idx2word)
                ground_truths.append(gt_tokens)

                # Convert model prediction sequence
                pred_tokens = ids_to_tokens(preds[i], code_idx2word)
                predictions.append(pred_tokens)

     # 5. COMPUTE METRICS
    print("Computing metrics...")
    
    total_samples = len(ground_truths)
    exact_match_count = 0
    valid_syntax_count = 0
    total_tokens = 0
    correct_tokens = 0

    # For NLTK corpus_bleu format requirements
    bleu_references = [[gt] for gt in ground_truths]
    bleu_hypotheses = predictions

    for gt, pred in zip(ground_truths, predictions):

        for gt_token, pred_token in zip_longest(
        gt,
        pred,
        fillvalue="<PAD>"
        ):
            total_tokens += 1

            if gt_token == pred_token:
                correct_tokens += 1

        if gt == pred:
            exact_match_count += 1

        predicted_code = " ".join(pred)

        try:
            ast.parse(predicted_code)
            valid_syntax_count += 1

        except Exception:
            pass

    # Calculations
    token_accuracy = (correct_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
    exact_match_pct = (exact_match_count / total_samples * 100)
    syntax_validity_pct = (valid_syntax_count / total_samples * 100)
    
    # (b) BLEU Score with smoothing to handle small sequences / 0 n-gram hits gracefully
    smooth_fn = SmoothingFunction().method1
    bleu_score = corpus_bleu(bleu_references, bleu_hypotheses, smoothing_function=smooth_fn) * 100

    # 6. SAVE PREDICTIONS CSV
    print(f"Saving outputs to {output_dir}...")
    
    df_predictions = pd.DataFrame({
        "Intent": raw_intents,
        "Ground Truth": [" ".join(gt) for gt in ground_truths],
        "Prediction": [" ".join(pred) for pred in predictions]
    })
    df_predictions.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # 7. SAVE METRICS TXT
    metrics = {
    "BLEU Score": bleu_score,
    "Exact Match (%)": exact_match_pct,
    "Token Accuracy (%)": token_accuracy,
    "Syntax Validity (%)": syntax_validity_pct,
    }

    with open(os.path.join(output_dir, "metrics.txt"),"w") as f:

        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")

    print("\nEvaluation Results")

    for key, value in metrics.items():
        print(f"{key:20}: {value:.2f}")

    
    print()

    print("=" * 60)
    print("Evaluation Completed")
    print("=" * 60)

    print(f"Samples             : {total_samples}")
    print(f"BLEU Score          : {bleu_score:.2f}")
    print(f"Exact Match         : {exact_match_pct:.2f}%")
    print(f"Token Accuracy      : {token_accuracy:.2f}%")
    print(f"Syntax Validity     : {syntax_validity_pct:.2f}%")

    print("=" * 60)


    print("Showing 10 random sample outputs:")
    print("---------------------------------")
    sample_indices = random.sample(range(total_samples), min(10, total_samples))
    
    with open(os.path.join(output_dir, "sample_outputs.txt"), "w") as f_samples:
        for idx in sample_indices:
            sample_str = (
                f"Intent      : {raw_intents[idx]}\n"
                f"Ground Truth: {' '.join(ground_truths[idx])}\n"
                f"Prediction  : {' '.join(predictions[idx])}\n"
                f"{'-'*40}\n"
            )
            print(sample_str)
            f_samples.write(sample_str)

if __name__ == "__main__":
    main()