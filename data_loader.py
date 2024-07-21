# file: data_loader.py

import pandas as pd

def load_data(ratings_path, anime_path):
    ratings = pd.read_csv(ratings_path)
    anime = pd.read_csv(anime_path)
    return ratings, anime
