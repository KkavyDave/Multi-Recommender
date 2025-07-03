import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

def load_data(csv_path):
    music_data = pd.read_csv(csv_path)
    
    # Remove duplicates by track_name + artists
    music_data = music_data.drop_duplicates(subset=['track_name', 'artists']).reset_index(drop=True)
    
    # Fill missing values
    music_data['track_name'] = music_data['track_name'].fillna('')
    music_data['artists'] = music_data['artists'].fillna('')
    music_data['track_genre'] = music_data['track_genre'].fillna('')
    
    # Build combined feature
    music_data['combined'] = music_data['track_name'] + ' ' + music_data['artists'] + ' ' + music_data['track_genre']
    
    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(music_data['combined'])
    
    return music_data, vectorizer, feature_vectors


def get_recommendations(song_name, music_data, vectorizer, feature_vectors, top_n=7):
    list_of_titles = music_data['track_name'].tolist()
    close_match = difflib.get_close_matches(song_name, list_of_titles)
    if not close_match:
        return [], None
    close_match = close_match[0]
    index = music_data[music_data.track_name == close_match].index.values[0]

    query_vec = feature_vectors[index]
    similarity = cosine_similarity(query_vec, feature_vectors).flatten()

    similar_indices = similarity.argsort()[::-1][1:top_n+1]

    recommendations = []
    for idx in similar_indices:
        title = music_data.iloc[idx]['track_name']
        artist = music_data.iloc[idx]['artists']
        genre = music_data.iloc[idx]['track_genre']
        score = round(similarity[idx] * 100, 2)
        recommendations.append((title, artist, genre, score))

    return recommendations, close_match
