import os
import yaml
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_enron, load_phishing
from src.data.cleaning import clean_dataset
from src.data.splits import create_stratified_splits

def main():
    # Load config
    with open('configs/datasets.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("=== Processing Enron Dataset ===")
    enron_cfg = config['datasets']['enron']
    if os.path.exists(enron_cfg['raw_path']):
        df_enron = load_enron(enron_cfg['raw_path'])
        df_enron = clean_dataset(df_enron)
        
        # Save processed data
        os.makedirs(os.path.dirname(enron_cfg['processed_path']), exist_ok=True)
        # Using parquet for efficient storage. Requires pyarrow or fastparquet.
        try:
            df_enron.to_parquet(enron_cfg['processed_path'])
        except ImportError:
            print("fastparquet/pyarrow not found. Saving as CSV instead.")
            enron_cfg['processed_path'] = enron_cfg['processed_path'].replace('.parquet', '.csv')
            df_enron.to_csv(enron_cfg['processed_path'], index=False)
            
        create_stratified_splits(df_enron, config, 'enron')
    else:
        print(f"Error: {enron_cfg['raw_path']} not found.")

    print("\n=== Processing Phishing Dataset ===")
    phishing_cfg = config['datasets']['phishing']
    if os.path.exists(phishing_cfg['raw_path']):
        df_phish = load_phishing(phishing_cfg['raw_path'])
        df_phish = clean_dataset(df_phish)
        
        os.makedirs(os.path.dirname(phishing_cfg['processed_path']), exist_ok=True)
        try:
            df_phish.to_parquet(phishing_cfg['processed_path'])
        except ImportError:
            print("fastparquet/pyarrow not found. Saving as CSV instead.")
            phishing_cfg['processed_path'] = phishing_cfg['processed_path'].replace('.parquet', '.csv')
            df_phish.to_csv(phishing_cfg['processed_path'], index=False)
            
        create_stratified_splits(df_phish, config, 'phishing')
    else:
        print(f"Error: {phishing_cfg['raw_path']} not found.")
        
    print("\nStage 1 completed successfully!")

if __name__ == "__main__":
    main()
