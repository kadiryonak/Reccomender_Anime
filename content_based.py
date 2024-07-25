import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

def prepare_content_based_matrix(anime, top_animes):
    anime = anime[anime['anime_id'].isin(top_animes['anime_id'])]
    anime.loc[:, 'genre'] = anime['genre'].fillna('')  # Using .loc to avoid SettingWithCopyWarning
    tfidf_matrix = anime['genre'].str.get_dummies(sep=',')
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return tfidf_matrix, cosine_sim

def get_indices(anime):
    return pd.Series(anime.index, index=anime['name']).drop_duplicates()

def content_based_recommendations(title, indices, cosine_sim, anime):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return anime.iloc[anime_indices][['name', 'rating', 'members']]
