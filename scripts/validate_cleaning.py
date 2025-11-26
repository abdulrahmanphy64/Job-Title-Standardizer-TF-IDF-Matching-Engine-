import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.cleaning.text_cleaner import TextCleaner

df = pd.read_csv("data/raw/job_titles_raw.csv")

df_first_ten = df["title"][:10]
print("Before Data cleaning")
print(df_first_ten)
print(type(df_first_ten))
cleaning_data = TextCleaner()
clean_title = []

for i in df_first_ten:
    clean_title.append(cleaning_data.clean(i))

print("After cleaning")
print(clean_title)






