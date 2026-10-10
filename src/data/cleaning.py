import pandas as pd
import hashlib

def normalize_text(text: str) -> str:
    """
    Basic normalization: lowercase and strip whitespace.
    """
    return text.lower().strip()

def deduplicate_emails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate emails by creating a SHA-256 hash of the normalized subject and body.
    """
    print(f"Initial count: {len(df)}")
    
    # Create a unified text for hashing
    combined_text = df['subject'].apply(normalize_text) + "|||" + df['body'].apply(normalize_text)
    
    # Hash it
    df['hash'] = combined_text.apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())
    
    # Deduplicate
    df_dedup = df.drop_duplicates(subset=['hash'])
    df_dedup = df_dedup.drop(columns=['hash']).reset_index(drop=True)
    
    print(f"Count after deduplication: {len(df_dedup)}")
    print(f"Removed {len(df) - len(df_dedup)} duplicates.")
    
    return df_dedup

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the dataset.
    """
    # 1. Deduplicate
    df_cleaned = deduplicate_emails(df)
    
    # 2. Additional cleaning could be added here (e.g. langdetect for English only)
    # For now, we'll keep it to deduplication as requested.
    
    return df_cleaned
