import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def make_recommendations(user_id, num_recommendations, movies_df, user_movie_ratings_original, movie_id_to_idx):
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    user_ratings = user_movie_ratings_original.loc[user_id]

    unrated_movies = user_ratings[user_ratings.isna()]

    unrated_movie_indices = [movie_id_to_idx[movie_id] for movie_id in unrated_movies.index if movie_id in movie_id_to_idx]

    user_unrated_predictions = predicted_ratings[user_id - 1, unrated_movie_indices]

    top_movie_indices = np.argsort(user_unrated_predictions)[-num_recommendations:][::-1]
    recommended_movie_ids = [unrated_movies.index[idx] for idx in top_movie_indices]

    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
    return recommended_movies['title'].tolist()


ratings_path = 'ml-latest-small/ratings.csv'
movies_path = 'ml-latest-small/movies.csv'

ratings_df = pd.read_csv(ratings_path)
movies_df = pd.read_csv(movies_path)


user_movie_ratings_original = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_ratings_for_svd = user_movie_ratings_original.fillna(0)
user_movie_rating_matrix = csr_matrix(user_movie_ratings_for_svd.values)


num_latent_features = 6
U, sigma, Vt = svds(user_movie_rating_matrix.toarray(), k=num_latent_features)
sigma = np.diag(sigma)


movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(user_movie_ratings_for_svd.columns)}


if __name__ == '__main__':
    user_id = int(input('UserId: '))
    num_recommendations = 10  # Number of recommendations
    recommendations = make_recommendations(user_id, num_recommendations, movies_df, user_movie_ratings_original, movie_id_to_idx)
    print(f"Top {num_recommendations} recommendations for User {user_id}:")
    for title in recommendations:
        print(title)