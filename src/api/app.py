import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.model.standardizer import Standardizer

app = FastAPI()

class Item(BaseModel):
    title : str

std = Standardizer()
std.load_index()

@app.post("/standardize")
def standardize(item : Item):
    result = std.find_best_match(item.title, threshold = 0.5)
    return result

