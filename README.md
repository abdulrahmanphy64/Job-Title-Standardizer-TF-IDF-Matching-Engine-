Job Title Standardizer (TF-IDF Matching Engine)

This project standardizes messy, inconsistent job titles into canonical, normalized forms using TF-IDF vectorization + cosine similarity.
It follows a clean ML engineering pipeline with modular structure, reproducible preprocessing, and a search-optimized vector index.

🚀 Project Objective

Real-world job titles are full of noise:
    1. “Sr. Data Scntst”
    2. “Data Scientist II”
    3. “Lead Data Scintist”
    4. “Data Scientist (Contract)”

All should map to a canonical form like:
data scientist

This project builds:

✔ A text-cleaning pipeline
✔ A TF-IDF vectorizer trained on cleaned titles
✔ A vector index for fast similarity search
✔ A Standardizer engine to return the best canonical match

🧹 1. Cleaning Pipeline

Located in:
src/cleaning/text_cleaner.py

Functions:
    1. Lowercasing
    2. Removing punctuation
    3. Expanding abbreviations (sr → senior, dev → developer, etc.)
    4. Normalizing multi-word job titles
    5. Removing extra noise

Example:
from src.cleaning.text_cleaner import TextCleaner

cleaner = TextCleaner()
cleaner.clean("Sr. Data Scntst (Contract)")
# → "senior data scientist"

