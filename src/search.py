import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm
from .config import SENTENCE_TRANSFORMER_MODEL, FAISS_INDEX_DIM

def generate_embeddings(texts: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """
    Generates normalized embeddings for a list of texts.
    """
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    return embeddings.astype(np.float32)

def l2_normalize(x: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    """
    L2 normalizes the embeddings.
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Creates a FAISS index for Inner Product search (Cosine Similarity on normalized vectors).
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def search_top_issues(
    df_data: pd.DataFrame,
    model: SentenceTransformer,
    index: faiss.Index,
    df_issues: pd.DataFrame,
    text_col: str = "Headline",
    issue_col: str = "issue_name",
    k: int = 3,
    batch_size: int = 64,
    search_chunk_size: int = 50000
) -> pd.DataFrame:
    """
    Searches for the top k matching issues for each headline in df_data.
    """
    texts = df_data[text_col].astype(str).fillna("").tolist()
    if not texts:
        return df_data

    issue_names_lookup = df_issues[issue_col].values
    
    n_samples = len(df_data)
    all_top_names = np.empty((n_samples, k), dtype=object)
    all_top_sims = np.empty((n_samples, k), dtype=np.float32)

    print(f"Processing {n_samples} headlines in chunks of {search_chunk_size}...")

    for start_idx in tqdm(range(0, n_samples, search_chunk_size), desc="Search & Map"):
        end_idx = min(start_idx + search_chunk_size, n_samples)
        batch_texts = texts[start_idx:end_idx]
        
        # 1. Encode
        embeddings = model.encode(
            batch_texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype(np.float32)
        
        # 2. Search
        D, I = index.search(embeddings, k)
        
        # 3. Map
        valid_mask = (I >= 0) & (I < len(issue_names_lookup))
        chunk_names = np.full(I.shape, None, dtype=object)
        chunk_names[valid_mask] = issue_names_lookup[I[valid_mask]]
        
        all_top_names[start_idx:end_idx] = chunk_names
        all_top_sims[start_idx:end_idx] = D

    df_out = df_data.copy()
    for r in range(k):
        df_out[f"issue_top{r+1}"] = all_top_names[:, r]
        df_out[f"sim_top{r+1}"] = all_top_sims[:, r].round(4)

    return df_out
