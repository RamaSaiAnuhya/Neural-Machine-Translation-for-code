import os
import pickle
import re
import pandas as pd
import ast

from sklearn.model_selection import train_test_split

def clean_text(text):
    # to lower case
    text = text.lower()
    # 2. Standardize quotation marks often found in Stack Overflow queries
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    # Replace every block of consecutive whitespace with a single whitespace
    text = re.sub(r'\s+', ' ', text)
    # Adds spacing around common programming characters mentioned in intent text
    text = re.sub(r'([+\-*/=(){}\[\].,:;<>!?])', r' \1 ', text)

    return text.strip()
 
def clean_code(code):
    # Strips comments and docstings from code snippets
    try:
        # Parse the code into an Abstract syntax tree
        parsed = ast.parse(code)

        # Remove docstrings
        for node in ast.walk(parsed):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if(node.body and isinstance(node.body[0], ast.Expr) and
                   isinstance(node.body[0].value, ast.Constant) and
                   isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)

        # Unparse back to clean string
        cleaned_code = ast.unparse(parsed)
        return cleaned_code
    
    except Exception:
        # If the code is just a snippet fragment and fails AST parsing
        # fall-back to standard string regex cleaning
        code_string = re.sub(r'(?m)^\s*#.*\n?', '', code)  
        return code_string
    
def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except Exception:
        return False
                
         
def main():

    # Load Dataset
    df = pd.read_json('data/raw/conala-paired-train.json', lines=True)
    df_mined = pd.read_json('data/raw/conola-mined-subset.json', lines=True)
    df_mined = df_mined[['intent', 'snippet']]

    original_size = len(df) + len(df_mined)

    # Preprocess the dataset
    print('Preprocessing the dataset...')
    df['rewritten_intent'] = df['rewritten_intent'].fillna(df['intent'])
    df = df[['rewritten_intent', 'snippet']]
    df.drop_duplicates(inplace = True)
    df_mined.dropna()
    df_mined.drop_duplicates(inplace=True)


    df = df.rename(columns={'rewritten_intent': 'intent'})

    # Clean the dataset
    print('Cleaning the dataset...')
    df['intent'] = df['intent'].apply(clean_text)
    df_mined['intent'] = df_mined['intent'].apply(clean_text)
    df['snippet'] = df['snippet'].apply(clean_code)
    df_mined['snippet'] = df_mined['snippet'].apply(clean_code)

    df = df[df['snippet'].apply(is_valid_python)]
    df_mined = df_mined[df_mined['snippet'].apply(is_valid_python)]

    df = df[(df['intent'].str.strip() != "") & (df['snippet'].str.strip() != "")]
    df_mined = df_mined[(df_mined['intent'].str.strip() != "") & (df_mined['snippet'].str.strip() != "")]

    df.reset_index(drop=True, inplace=True)
    df_mined.reset_index(drop=True, inplace=True)

    # Dataset statistics before and after cleaning
    print(f"Original samples : {original_size}")
    print(f"Remaining samples: {len(df) + len(df_mined)}")
    print(f"Removed samples  : {original_size - (len(df) + len(df_mined))}")

    # Split the dataset into training and validation sets
    print('Splitting the dataset into training and validation sets...')
    train_df, val_df = train_test_split(
        df, test_size=0.1, shuffle=True, random_state=42
    )

    # Stack them vertically
    train_df = pd.concat([train_df, df_mined], ignore_index=True)

    print('Number of rows in the training set:', len(train_df))
    print('Number of rows in the validation set:', len(val_df))
 
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok = True)

    output_train_path = os.path.join(output_dir, 'train.pkl')
    output_val_path = os.path.join(output_dir, 'val.pkl')

    # Save the processed training and validation sets 
    print(f'Saving processed training set to {output_train_path}...')
    with open(output_train_path, 'wb') as f:
        pickle.dump(train_df, f)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index = False)
    
    print(f'Saving processed validation set to {output_val_path}...')
    with open(output_val_path, 'wb') as f:
        pickle.dump(val_df, f)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index = False)

    # Print Statistics
    print('Average intent length: ', df['intent'].str.split().apply(len).mean())
    print('Average snippet length: ', df['snippet'].str.split().apply(len).mean())

    print('Max intent length: ', df['intent'].str.split().apply(len).max())
    print('Max snippet length: ', df['snippet'].str.split().apply(len).max())

if __name__ == '__main__':
    main()

    

