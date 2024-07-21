# file: preprocess.py
import pandas as pd

def preprocess_data(ratings, anime):
    anime.dropna(subset=['genre', 'type', 'rating'], inplace=True)
    ratings.drop_duplicates(subset=['user_id', 'anime_id'], keep='last', inplace=True)
    ratings['rating'] = ratings['rating'].astype(float)
    return ratings, anime

def filter_top_animes(anime, ratings, top_n=4000, max_users=4000):
    top_animes = anime.sort_values(by='rating', ascending=False).head(top_n)
    top_anime_ids = top_animes['anime_id'].values
    filtered_ratings = ratings[ratings['anime_id'].isin(top_anime_ids)]
    unique_users = filtered_ratings['user_id'].unique()[:max_users]
    reduced_ratings = filtered_ratings[filtered_ratings['user_id'].isin(unique_users)]
    return top_animes, reduced_ratings