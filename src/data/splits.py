import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_stratified_splits(df: pd.DataFrame, config: dict, dataset_name: str):
    """
    Creates stratified splits (train, calibration, validation, test) 
    and saves the INDICES to disk to ensure reproducibility.
    """
    splits_cfg = config['splits']
    seed = config['seed']
    out_dir = splits_cfg['split_indices_dir']
    
    os.makedirs(out_dir, exist_ok=True)
    
    # We'll split sequentially:
    # 1. Test set
    # 2. Validation set
    # 3. Calibration set
    # 4. Train set (remainder)
    
    test_ratio = splits_cfg['test']
    val_ratio = splits_cfg['validation']
    cal_ratio = splits_cfg['calibration']
    train_ratio = splits_cfg['train']
    
    # Verify ratios sum to ~1
    assert abs((test_ratio + val_ratio + cal_ratio + train_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1"
    
    # Indices array
    indices = np.arange(len(df))
    labels = df['label'].values
    
    # Extract test
    idx_temp, idx_test, y_temp, _ = train_test_split(
        indices, labels, 
        test_size=test_ratio, 
        random_state=seed, 
        stratify=labels
    )
    
    # Relative ratio for validation out of remaining
    rel_val_ratio = val_ratio / (train_ratio + cal_ratio + val_ratio)
    idx_temp2, idx_val, y_temp2, _ = train_test_split(
        idx_temp, y_temp, 
        test_size=rel_val_ratio, 
        random_state=seed, 
        stratify=y_temp
    )
    
    # Relative ratio for calibration out of remaining
    rel_cal_ratio = cal_ratio / (train_ratio + cal_ratio)
    idx_train, idx_cal, _, _ = train_test_split(
        idx_temp2, y_temp2,
        test_size=rel_cal_ratio,
        random_state=seed,
        stratify=y_temp2
    )
    
    # Save indices to disk
    np.save(os.path.join(out_dir, f'{dataset_name}_train.npy'), idx_train)
    np.save(os.path.join(out_dir, f'{dataset_name}_cal.npy'), idx_cal)
    np.save(os.path.join(out_dir, f'{dataset_name}_val.npy'), idx_val)
    np.save(os.path.join(out_dir, f'{dataset_name}_test.npy'), idx_test)
    
    print(f"Splits saved for {dataset_name}:")
    print(f"  Train: {len(idx_train)} ({len(idx_train)/len(df):.1%})")
    print(f"  Cal:   {len(idx_cal)} ({len(idx_cal)/len(df):.1%})")
    print(f"  Val:   {len(idx_val)} ({len(idx_val)/len(df):.1%})")
    print(f"  Test:  {len(idx_test)} ({len(idx_test)/len(df):.1%})")
    
    return {
        'train': idx_train,
        'cal': idx_cal,
        'val': idx_val,
        'test': idx_test
    }
