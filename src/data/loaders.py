import pandas as pd
import numpy as np

def load_enron(filepath: str) -> pd.DataFrame:
    """
    Load the Enron dataset from CSV.
    Expected columns: subject, body, label (0=ham, 1=spam)
    """
    print(f"Loading Enron dataset from {filepath}...")
    # Some rows might have parsing issues, so we use on_bad_lines='skip' or error_bad_lines=False for older pandas
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(filepath, error_bad_lines=False)
        
    if 'subject' not in df.columns or 'body' not in df.columns or 'label' not in df.columns:
        raise ValueError("Enron dataset must contain 'subject', 'body', and 'label' columns.")
        
    df = df[['subject', 'body', 'label']]
    
    # Ensure strings
    df['subject'] = df['subject'].fillna('').astype(str)
    df['body'] = df['body'].fillna('').astype(str)
    
    return df

def load_phishing(filepath: str) -> pd.DataFrame:
    """
    Load the Phishing dataset from CSV.
    Expected columns: text_combined, label (0=ham, 1=phishing)
    """
    print(f"Loading Phishing dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(filepath, error_bad_lines=False)
        
    if 'text_combined' not in df.columns or 'label' not in df.columns:
        raise ValueError("Phishing dataset must contain 'text_combined' and 'label' columns.")
        
    # Standardize column names to subject, body, label
    # Phishing dataset doesn't explicitly separate subject, so we'll put it all in body
    df['subject'] = ""
    df['body'] = df['text_combined'].fillna('').astype(str)
    
    df = df[['subject', 'body', 'label']]
    return df
