import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample movie ratings data
movie_ratings = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'movie_id': [101, 102, 103, 101, 104, 102, 103, 105, 101, 105],
    'rating': [5, 4, 3, 4, 2, 5, 3, 4, 1, 5]
}
ratings_df = pd.DataFrame(movie_ratings)

# Movie titles
movies = {
    'movie_id': [101, 102, 103, 104, 105],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']
}
movies_df = pd.DataFrame(movies)

# Merge datasets
data = pd.merge(ratings_df, movies_df, on='movie_id')
print("Merged Data:")
print(data.head())

# Create a user-item interaction matrix (users as rows, movies as columns)
user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
print("\nUser-Item Interaction Matrix:")
print(user_movie_matrix)

# Compute similarity between users using cosine similarity
user_similarity = cosine_similarity(user_movie_matrix)
similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
print("\nUser Similarity Matrix:")
print(similarity_df)

# Function to recommend movies for a user
def recommend_movies(user_id, user_movie_matrix, similarity_df, top_n=3):
    # Get similar users sorted by similarity score
    similar_users = similarity_df[user_id].sort_values(ascending=False).index[1:]  # Skip the user itself
    recommended_movies = pd.Series()

    # Loop through similar users and gather their highly rated movies
    for similar_user in similar_users:
        movies_watched_by_similar_user = user_movie_matrix.loc[similar_user]
        high_rated_movies = movies_watched_by_similar_user[movies_watched_by_similar_user > 3]  # Movies rated 4+
        recommended_movies = recommended_movies.append(high_rated_movies)

        if len(recommended_movies) >= top_n:
            break

    # Return top N recommended movie titles
    return recommended_movies.index[:top_n]

# Example: Recommend movies for user 1
user_id = 1
recommendations = recommend_movies(user_id, user_movie_matrix, similarity_df)
print(f"\nRecommended Movies for User {user_id}: {recommendations}")
