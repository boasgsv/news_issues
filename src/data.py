import pathlib
import pandas as pd
import os
import kagglehub
from .config import DATASET_ID_NYT, DATASET_ID_LARGE

def download_file_if_not_exists(path: pathlib.Path, gdown_id: str = None):
    """
    Downloads a file if it does not exist (helper for IPTC issues).
    """
    if path.exists():
        print(f"File {path} already exists.")
        return

    print(f"Downloading {path}...")
    if gdown_id:
        import gdown
        gdown.download(id=gdown_id, output=str(path), quiet=False)
    else:
        print("No download source provided.")

def load_iptc_issues(path: pathlib.Path, gdown_id: str = None) -> pd.DataFrame:
    """
    Loads IPTC issues taxonomy from a parquet file.
    """
    if not path.exists() and gdown_id:
        download_file_if_not_exists(path, gdown_id=gdown_id)
    return pd.read_parquet(path)

def load_news_data(dataset_name: str = 'nyt') -> pd.DataFrame:
    """
    Loads news headlines from Kaggle dataset using kagglehub.
    Returns a DataFrame with at least a 'Headline' column.
    """
    if dataset_name == 'nyt':
        dataset_id = DATASET_ID_NYT
        print(f"Downloading/Loading dataset: {dataset_id}")
        path = kagglehub.dataset_download(dataset_id)
        
        # Find the parquet file
        parquet_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_file = os.path.join(root, file)
                    break
            if parquet_file:
                break
                
        if not parquet_file:
            raise FileNotFoundError("No parquet file found in the downloaded dataset.")
            
        print(f"Loading data from {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        # Rename 'title' to 'Headline'
        if 'title' in df.columns:
            df = df.rename(columns={'title': 'Headline'})
            
    elif dataset_name == 'large':
        dataset_id = DATASET_ID_LARGE
        print(f"Downloading/Loading dataset: {dataset_id}")
        path = kagglehub.dataset_download(dataset_id)
        
        # Find the csv file
        csv_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith("headlines.csv"):
                    csv_file = os.path.join(root, file)
                    break
            if csv_file:
                break
        
        if not csv_file:
            raise FileNotFoundError("headlines.csv not found in the downloaded dataset.")
            
        print(f"Loading data from {csv_file}...")
        # This dataset is large, so we might want to be careful, but for now load it.
        # It has 'Headline' column already.
        df = pd.read_csv(csv_file)
        
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    # Ensure Headline exists
    if 'Headline' not in df.columns:
        raise ValueError(f"Dataset does not contain 'Headline' column. Columns: {df.columns.tolist()}")
        
    return df
