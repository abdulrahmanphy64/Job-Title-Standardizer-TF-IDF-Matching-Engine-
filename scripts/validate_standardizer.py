import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.model.standardizer import Standardizer

std = Standardizer()
print(std.build_index())
print(std.load_index())

res = std.find_best_match("Sr. ML Engg", threshold= 0.5)

print(res)