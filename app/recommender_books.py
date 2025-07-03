import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

def load_data(csv_path):
    # Load books dataset
    books_data = pd.read_csv(csv_path)
    # Combine title and authors into a single text feature
    books_data['combined'] = books_data['title'] + ' ' + books_data['authors']
    books_data['combined'] = books_data['combined'].fillna('')
    # Build TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(books_data['combined'])
    # Compute cosine similarity
    similarity = cosine_similarity(feature_vectors)
    return books_data, similarity

def get_recommendations(book_name, books_data, similarity, top_n=7):
    list_of_titles = books_data['title'].tolist()
    close_match = difflib.get_close_matches(book_name, list_of_titles)
    if not close_match:
        return [], None
    close_match = close_match[0]
    index = books_data[books_data.title == close_match].index.values[0]
    similarity_scores = list(enumerate(similarity[index]))
    sorted_books = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]
    recommendations = []
    for i, (idx, score) in enumerate(sorted_books[:top_n]):
        title = books_data.iloc[idx]['title']
        author = books_data.iloc[idx]['authors']
        recommendations.append((title, author, round(score*100, 2)))
    return recommendations, close_match
