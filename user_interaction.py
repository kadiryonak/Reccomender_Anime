# file: user_interaction.py

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel  # Gerekli import

def generate_comparison_questions(anime_df, num_questions=10):
    comparisons = []
    for _ in range(num_questions):
        pair = anime_df.sample(n=2)
        comparisons.append(pair)
    return comparisons

def get_user_responses(questions):
    user_responses = []
    for i, pair in enumerate(questions):
        while True:
            print(f"Question {i + 1}:")
            print(f"1: {pair.iloc[0]['name']} (ID: {pair.iloc[0]['anime_id']}) vs 2: {pair.iloc[1]['name']} (ID: {pair.iloc[1]['anime_id']})")
            response = input("Which anime do you prefer? (1 or 2): ")
            if response in ['1', '2']:
                break
            else:
                print("Invalid input. Please enter 1 or 2.")
        user_responses.append((pair.iloc[0] if response == '1' else pair.iloc[1]))
    return pd.DataFrame(user_responses)

def personalized_recommendations(user_profile, tfidf_matrix, anime, cosine_sim):
    sim_scores = linear_kernel(user_profile.values.reshape(1, -1), tfidf_matrix).flatten()
    sim_scores = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:20]
    anime_indices = [i[0] for i in sim_scores]
    return anime.iloc[anime_indices][['name', 'rating', 'members']]