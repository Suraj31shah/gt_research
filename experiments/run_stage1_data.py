import os
import yaml
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_enron, load_phishing, get_active_dataset
from src.data.cleaning import clean_dataset
from src.data.splits import create_stratified_splits

def process_single_dataset(name: str, cfg: dict, config: dict, loader_fn):
    print(f"\n=== Processing {name.capitalize()} Dataset ===")
    if not os.path.exists(cfg['raw_path']):
        print(f"Error: {cfg['raw_path']} not found.")
        return

    df = loader_fn(cfg['raw_path'])
    df = clean_dataset(df)
    
    out_path = cfg['processed_path']
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if out_path.endswith('.parquet'):
        try:
            df.to_parquet(out_path)
        except ImportError:
            print("fastparquet/pyarrow not found. Saving as CSV instead.")
            out_path = out_path.replace('.parquet', '.csv')
            df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
        
    create_stratified_splits(df, config, name)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Process and Split Dataset")
    parser.add_argument('--dataset', default=None, choices=['enron', 'phishing', 'all'],
                        help="Dataset to process (default: active_dataset in config, or all)")
    args = parser.parse_args()

    # Load config
    with open('configs/datasets.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    target = args.dataset or config.get('active_dataset', 'phishing')
    
    if target in ('enron', 'all'):
        process_single_dataset('enron', config['datasets']['enron'], config, load_enron)
        
    if target in ('phishing', 'all'):
        process_single_dataset('phishing', config['datasets']['phishing'], config, load_phishing)
        
    print("\nStage 1 completed successfully!")

if __name__ == "__main__":
    main()
