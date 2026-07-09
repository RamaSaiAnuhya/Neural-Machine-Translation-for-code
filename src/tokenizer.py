import os
import io
import pickle
import tokenize
from collections import Counter

class CustomTokenizer:
    def __init__(self):
        pass
        
    def tokenize_intent(self, text):
        # Tokenize the intent text
        tokens = text.split()
        return tokens
    
    def tokenize_code(self, code):
        tokens = []
        try:
            token_generator = tokenize.generate_tokens(io.StringIO(code).readline)
            for tok in token_generator:
                if tok.type in (
                    tokenize.ENCODING,
                    tokenize.ENDMARKER,
                    tokenize.NEWLINE,
                    tokenize.NL,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                ):
                    continue
                tokens.append(tok.string)
        except tokenize.TokenError:
            return code.split()

        return tokens
    
class VocabularyBuilder:
    def __init__(self):
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

    def build_vocab(self, tokenized_texts, min_freq = 1):
        input_counter = Counter()
        for tokens in tokenized_texts:
            input_counter.update(tokens)

        word2idx = {}

        for token in self.special_tokens:
            word2idx[token] = len(word2idx)
        for word, freq in input_counter.most_common():
            if freq >= min_freq:
                word2idx[word] = len(word2idx)

        idx2word = {
            v:k for k, v in word2idx.items()
        }

        return word2idx, idx2word

class IdConvertor:
    def __init__(self):
        pass

    def to_ids(self, tokens, word2idx):
        ids = [
            word2idx.get(token, word2idx["<UNK>"])
            for token in tokens
        ]

        return (
            [word2idx["<SOS>"]]
            + ids +
            [word2idx["<EOS>"]]
        )

def main():
    
    # Load the cleaned data from preprocess.py
    train_input_path = os.path.join("data/processed", "train.pkl")
    val_input_path = os.path.join("data/processed", "val.pkl")

    if not os.path.exists(train_input_path) or not os.path.exists(val_input_path):
        raise FileNotFoundError("Processed data not found. Please run preprocess.py first")
    else:
        with open(train_input_path, 'rb') as f:
            train_df = pickle.load(f)
        with open(val_input_path, 'rb') as f:
            val_df = pickle.load(f)

    # Initialize the tokenizer
    tokenizer = CustomTokenizer()

    print("Tokenizing Text...")
    tokenized_train_text = [
        tokenizer.tokenize_intent(text) for text in train_df['rewritten_intent']
    ]
    tokenized_val_text = [
        tokenizer.tokenize_intent(text) for text in val_df['rewritten_intent']
    ]


    print('Tokenizing code...')
    tokenized_train_code = [
        tokenizer.tokenize_code(code) for code in train_df['snippet']
    ]
    tokenized_val_code = [
        tokenizer.tokenize_code(code) for code in val_df['snippet']
    ]

    # Initialize the vocabulary builder
    vocab_builder = VocabularyBuilder()

    print("Building Text Vocabulary...")
    text_word2idx, text_idx2word = vocab_builder.build_vocab(tokenized_train_text, min_freq = 1)

    print('Building code Vocabulary...')
    code_word2idx, code_idx2word = vocab_builder.build_vocab(tokenized_train_code, min_freq = 1)

    # Vocabulary Statistics
    print(f'Text Vocabulary size: {len(text_word2idx)}')
    print(f'Code Vocabulary size: {len(code_word2idx)}')

    with open('config.py', 'w') as f:
        f.write(f'text_vocab_size = {len(text_word2idx)}\n')
        f.write(f'code_vocab_size = {len(code_word2idx)}\n')

    # Initialize the ToIds convertor
    id_convertor = IdConvertor()

    print("Converting Text Tokens to IDs...") 
    train_text_ids = [
        id_convertor.to_ids(tokens, text_word2idx) for tokens in tokenized_train_text
    ]
    val_text_ids = [    
        id_convertor.to_ids(tokens, text_word2idx) for tokens in tokenized_val_text
    ]

    print("Converting Code Tokens to IDs...")
    train_code_ids = [
        id_convertor.to_ids(tokens, code_word2idx) for tokens in tokenized_train_code
    ]
    val_code_ids = [
        id_convertor.to_ids(tokens, code_word2idx) for tokens in tokenized_val_code
    ]

    # Maximum sequence lengths
    print("Maximum intent length in training data:", max(len(x) for x in train_text_ids))
    print("Maximum snippet length in training data:", max(len(x) for x in train_code_ids))

    output_dir = 'data/vocabulary'
    os.makedirs(output_dir, exist_ok = True)

    # Path for saving the vocabulary
    text_word2idx_path = os.path.join(output_dir, 'text_word2idx.pkl')
    text_idx2word_path = os.path.join(output_dir, 'text_idx2word.pkl')

    code_word2idx_path = os.path.join(output_dir, 'code_word2idx.pkl')
    code_idx2word_path = os.path.join(output_dir, 'code_idx2word.pkl')

    # Path for saving the tokenized data
    train_text_ids_path = os.path.join(output_dir, 'train_text_ids.pkl')
    train_code_ids_path = os.path.join(output_dir, 'train_code_ids.pkl')
    val_text_ids_path = os.path.join(output_dir, 'val_text_ids.pkl')
    val_code_ids_path = os.path.join(output_dir, 'val_code_ids.pkl')


    # Save the processed training and validation sets 
    print(f'Saving Vocabularies to {output_dir}...')
    with open(text_word2idx_path, 'wb') as f:
        pickle.dump(text_word2idx, f)
    with open(text_idx2word_path, 'wb') as f:
        pickle.dump(text_idx2word, f)
    with open(code_word2idx_path, 'wb') as f:
        pickle.dump(code_word2idx, f)
    with open(code_idx2word_path, 'wb') as f:
        pickle.dump(code_idx2word, f)

    print(f'Saving tokenized training and validation sets to {output_dir}')
    with open(train_text_ids_path, 'wb') as f:
        pickle.dump(train_text_ids, f)
    with open(train_code_ids_path, 'wb') as f:
        pickle.dump(train_code_ids, f)
    with open(val_text_ids_path, 'wb') as f:
        pickle.dump(val_text_ids, f)
    with open(val_code_ids_path, 'wb') as f:
        pickle.dump(val_code_ids, f)  

if __name__ == '__main__':
    main()