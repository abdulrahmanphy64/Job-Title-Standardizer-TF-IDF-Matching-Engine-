import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.features.vectorizer import TitleVectorizer

if __name__ == "__main__":
    tv = TitleVectorizer()
    matrix = tv.build_vectorier()
    print("TF-IDF vectorizer built successfully")
    print("Shap:", matrix.shape)