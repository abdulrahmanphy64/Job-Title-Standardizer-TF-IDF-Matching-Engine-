import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.cleaning.text_cleaner import TextCleaner

df = pd.read_csv("data/raw/job_titles_raw.csv")

def process_title():
    cleaning_data = TextCleaner()
    df_copy = df.copy()
    df_copy.rename(columns={"title" : "title_raw"}, inplace=True)
    clean_title = []

    for i in df_copy["title_raw"]:
        clean_title.append(cleaning_data.clean(i))

    df_copy['clean_title'] = clean_title

    if not os.path.exists("data/processed/job_titles_cleaned.csv"):
        os.makedirs("data/processed/job_titles_cleaned.csv")
        df_copy.to_csv("data/processed/job_titles_cleaned.csv")

    print(f"Number of rows in raw csv {df.shape[0]}")
    print(f"Number of rows in cleaned csv {df_copy.shape[0]}")

if __name__ == "__main__":
    process_title()
