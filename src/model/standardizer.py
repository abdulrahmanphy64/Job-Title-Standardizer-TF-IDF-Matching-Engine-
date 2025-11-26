import os
import sys
import json 
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.cleaning.text_cleaner import TextCleaner

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


class Standardizer:
    def __init__(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.model_dir = os.path.join(base, "model")
        self.index_vec_path = os.path.join(self.model_dir, "index_vectors.npz")
        self.index_meta_path = os.path.join(self.model_dir, "index_meta.json")
        self.tfidf_path = os.path.join(self.model_dir, "tfidf.pkl")
        self.cleaner = TextCleaner()
    
    def build_index(self):
        """
            Build TF-IDF index from cleaned job titles and save:
            - sparse vectors
            - metadata list (canonical titles)
        """
        # Load clean data
        df = pd.read_csv("data/processed/job_titles_cleaned.csv")
        
        # Extract unique titles
        unique_titles = df['clean_title'].dropna().unique().tolist()
        
        # Load the tf-idf model
        if not os.path.exists(self.tfidf_path):
            raise FileNotFoundError("tfidf.pkl not found in model directory")
        
        self.tfidf = joblib.load(self.tfidf_path)

        vectors = self.tfidf.transform(unique_titles)

        os.makedirs(self.model_dir, exist_ok=True)

        sparse.save_npz(self.index_vec_path, vectors)

        with open(self.index_meta_path, "w") as f:
            json.dump(unique_titles, f)

        return "Index built successfully"
    

    def load_index(self):
        if not os.path.exists(self.index_vec_path):
            raise FileNotFoundError("Index vectors not found. Run build_index() first.")
        
        if not os.path.exists(self.index_meta_path):
            raise FileNotFoundError("Index metadata not found. Run build_index() first.")
        
        if not os.path.exists(self.tfidf_path):
            raise FileNotFoundError("TF-IDF model missing. Rebuild vectorizer")

        self.index_vectors = sparse.load_npz(self.index_vec_path)

        self.index_vectors = normalize(self.index_vectors, axis = 1)

        with open(self.index_meta_path, 'r') as f:
            self.meta = json.load(f)

        self.tfidf = joblib.load(self.tfidf_path)

        if self.index_vectors.shape[0] != len(self.meta):
            raise ValueError("Index vectors and metadata size mismatch - rebuild index.")
        
        return "Index loaded successfully"



    def find_best_match(self, raw_text : str, threshold : float = 0.5):
        if not hasattr(self, "index_vectors") or not hasattr(self, "meta"):
            raise RuntimeError("Index not loaded. Call load_index() first.")
        
        
        cleaned = self.cleaner.clean(raw_text or "")

        if cleaned == "":
            return {"input" : raw_text, "cleaned" : "", "canonical" : "", "score" : 0.0}
        
        try:
            exact_idx = self.meta.index(cleaned)
            return {"input" : raw_text, "cleaned" : cleaned, "canonical" : self.meta[exact_idx], "score" : 1.0}
        except ValueError:
            pass

        q_vec = self.tfidf.transform([cleaned])

        if q_vec.nnz == 0:
            return {
                "input" : raw_text,
                "cleaned" : cleaned,
                "canonical" : cleaned,
                "score" : 0.0

            }

        q_vec = normalize(q_vec, axis=1)

        sims = (self.index_vectors.dot(q_vec.T)).toarray().ravel() if sparse.issparse(self.index_vectors)else (self.index_vectors @q_vec.T).ravel()

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= threshold:
            canonical = self.meta[best_idx]
        else:
            canonical = self.cleaned

        return {
            "input" : raw_text,
            "cleaned" : self.cleaned,
            "canonical" : canonical,
            "score" : best_score
        }



