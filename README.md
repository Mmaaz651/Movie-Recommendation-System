# ML-Movie-Recommendation-System

A movie recommendation system built using machine learning techniques. The system leverages **cosine similarity** to recommend movies to users based on their ratings and preferences. It calculates the similarity between users and provides movie recommendations using collaborative filtering.

## Project Overview

This project uses movie ratings data to create a recommendation system that suggests movies to users based on their preferences and the preferences of similar users. The system includes the following steps:

1. **Data Collection & Preprocessing**: Collects user ratings and movie data and merges them into a consolidated dataset.
2. **User-Item Interaction Matrix**: Creates a matrix representing user ratings for each movie.
3. **Cosine Similarity Calculation**: Computes the similarity between users based on their ratings.
4. **Movie Recommendation**: Suggests movies for a user based on the ratings of similar users.

## Libraries Used

- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `sklearn` for calculating cosine similarity.

## How It Works

1. **Data**: The dataset includes user ratings for different movies.
2. **User-Item Interaction Matrix**: A matrix is created where rows represent users and columns represent movies.
3. **Cosine Similarity**: The cosine similarity metric is used to calculate the similarity between users based on their movie ratings.
4. **Recommendation**: A function recommends movies by considering the ratings of similar users, prioritizing highly-rated movies.
