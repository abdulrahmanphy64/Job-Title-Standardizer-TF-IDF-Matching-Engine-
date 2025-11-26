import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)


class TitleVectorizer:
    def __init__(self):
        self.vectorizer_path = "src/model/tfidf.pkl"
        os.makedirs("src/model", exist_ok=True)

    def load_data(self):
        df = pd.read_csv("data/processed/job_titles_cleaned.csv")
        return df["clean_title"].to_list()
    
    def build_vectorier(self):
        titles = self.load_data()

        tfidf = TfidfVectorizer(
            ngram_range=(1,2),
            min_df = 2,
            max_features=5000
        )

        matrix = tfidf.fit_transform(titles)

        joblib.dump(tfidf, self.vectorizer_path)

        return matrix
    
    

    


    


