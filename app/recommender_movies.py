import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

def load_data(csv_path):
    movies_data = pd.read_csv(csv_path)
    selected_features = ['genres','keywords','tagline','cast','director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return movies_data, similarity

def get_recommendations(movie_name, movies_data, similarity, top_n=7):
    list_of_titles = movies_data['title'].tolist()
    close_match = difflib.get_close_matches(movie_name, list_of_titles)
    if not close_match:
        return [], None
    close_match = close_match[0]
    index = movies_data[movies_data.title == close_match].index.values[0]
    similarity_scores = list(enumerate(similarity[index]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]
    recommendations = []
    for i, (idx, score) in enumerate(sorted_movies[:top_n]):
        title = movies_data.iloc[idx]['title']
        recommendations.append((title, round(score*100, 2)))
    return recommendations, close_match
