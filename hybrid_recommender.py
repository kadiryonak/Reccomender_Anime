# file: hybrid_recommender.py

import pandas as pd
from content_based import content_based_recommendations

def hybrid_recommendations(user_id, preds_df, reduced_ratings, anime, indices, cosine_sim, title, num_recommendations=5):
    content_recs = content_based_recommendations(title, indices, cosine_sim, anime)
    user_row_number = user_id - 1
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    user_data = reduced_ratings[reduced_ratings.user_id == user_id]
    recommendations = pd.DataFrame(sorted_user_predictions).reset_index()
    recommendations.columns = ['anime_id', 'predicted_rating']
    recommendations = recommendations[~recommendations['anime_id'].isin(user_data['anime_id'])]
    recommendations = recommendations.merge(anime, on='anime_id')
    combined_recommendations = recommendations[recommendations['name'].isin(content_recs['name'])]
    final_recommendations = combined_recommendations[['anime_id', 'name', 'predicted_rating', 'rating', 'members']].head(num_recommendations)
    return final_recommendations
