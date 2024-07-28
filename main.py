# file: main.py
import pandas as pd
from data_loader import load_data
from preprocess import preprocess_data, filter_top_animes
from collaborative_filtering import create_user_item_matrix, train_svd, predict_ratings
from content_based import prepare_content_based_matrix, get_indices
from user_interaction import generate_comparison_questions, get_user_responses, personalized_recommendations
from hybrid_recommender import hybrid_recommendations

def main():
    ratings_path = r"C:\Users\kadir\Downloads\archive\rating.csv"
    anime_path = r"C:\Users\kadir\Downloads\archive\anime.csv"

    ratings, anime = load_data(ratings_path, anime_path)
    ratings, anime = preprocess_data(ratings, anime)

    top_animes, reduced_ratings = filter_top_animes(anime, ratings)

    R = create_user_item_matrix(reduced_ratings)
    U, sigma, Vt = train_svd(R, k=100)  # k değerini artırdık
    preds_df = predict_ratings(U, sigma, Vt)

    # NaN değerlerini ortalama tahmin ile dolduruyoruz
    preds_df.fillna(preds_df.mean(), inplace=True)

    tfidf_matrix, cosine_sim = prepare_content_based_matrix(anime, top_animes)
    indices = get_indices(anime)

    random_20_anime = top_animes.sample(n=20)
    questions = generate_comparison_questions(random_20_anime)
    user_responses_df = get_user_responses(questions)

    user_genre_profile = user_responses_df['genre'].str.get_dummies(sep=',').mean()

    missing_cols = set(tfidf_matrix.columns) - set(user_genre_profile.index)
    for col in missing_cols:
        user_genre_profile[col] = 0
    user_genre_profile = user_genre_profile[tfidf_matrix.columns]

    personalized_recs = personalized_recommendations(user_genre_profile, tfidf_matrix, anime, cosine_sim)

    user_id = 1
    title = random_20_anime.iloc[0]['name']
    hybrid_recs = hybrid_recommendations(user_id, preds_df, reduced_ratings, anime, indices, cosine_sim, title, 10)  # 5 yerine 10 öneri yapıyoruz

    final_recommendations = pd.concat([personalized_recs.head(10), hybrid_recs]).drop_duplicates().head(10)  # 5 yerine 10 öneri
    print("Final Recommendations:")
    print(final_recommendations)

if __name__ == "__main__":
    main()
