import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from app.recommender_books import load_data as load_books, get_recommendations as get_book_recs
from app.recommender_movies import load_data as load_movies, get_recommendations as get_movie_recs
from app.recommender_music import load_data as load_music, get_recommendations as get_music_recs

# Use session_state to keep track of the current page
if 'page' not in st.session_state:
    st.session_state['page'] = None

st.set_page_config(page_title="Multi Recommender", page_icon="🎬")

# --------------------------------
# 📚 BOOK PAGE
def show_books():
    st.header("Book Recommendation System")
    books_data, similarity_books = load_books("data/books.csv")
    book_name = st.text_input("Enter your favourite book:")
    top_n = st.slider("Number of recommendations", 3, 15, 7)

    if book_name:
        recs, matched_name = get_book_recs(book_name, books_data, similarity_books, top_n)
        if matched_name:
            if recs[0][2] < 20:
                st.warning("Sorry, we couldn't find close matches. Please try entering a different book title.")
            else:
                df = pd.DataFrame(recs, columns=["Book", "Author", "Similarity (%)"])
                df.index += 1
                st.table(df)
                fig, ax = plt.subplots()
                ax.barh(df["Book"][::-1], df["Similarity (%)"][::-1], color="mediumseagreen")
                ax.set_xlabel("Similarity (%)")
                st.pyplot(fig)
        else:
            st.warning("No close match found. Please check the book title.")

    st.button("⬅️ Back to Home", on_click=lambda: st.session_state.update({'page': None}))


# --------------------------------
# 🎬 MOVIE PAGE
def show_movies():
    st.header("Movie Recommendation System")
    movies_data, similarity_movies = load_movies("data/movies.csv")
    movie_name = st.text_input("Enter your favourite movie:")
    top_n = st.slider("Number of recommendations", 3, 15, 7)

    if movie_name:
        recs, matched_name = get_movie_recs(movie_name, movies_data, similarity_movies, top_n)
        if matched_name:
            if recs[0][1] < 20:
                st.warning("Sorry, we couldn't find close matches. Please try entering a different movie.")
            else:
                df = pd.DataFrame(recs, columns=["Movie", "Similarity (%)"])
                df.index += 1
                st.table(df)
                fig, ax = plt.subplots()
                ax.barh(df["Movie"][::-1], df["Similarity (%)"][::-1], color="skyblue")
                ax.set_xlabel("Similarity (%)")
                st.pyplot(fig)
        else:
            st.warning("No close match found. Please check the movie name.")

    st.button("⬅️ Back to Home", on_click=lambda: st.session_state.update({'page': None}))


# --------------------------------
# 🎵 MUSIC PAGE
def show_music():
    st.header("Music Recommendation System")
    music_data, vectorizer_music, feature_vectors_music = load_music("data/SpotifyTracks.csv")
    song_name = st.text_input("Enter your favourite song:")
    top_n = st.slider("Number of recommendations", 3, 15, 7)

    if song_name:
        recs, matched_name = get_music_recs(song_name, music_data, vectorizer_music, feature_vectors_music, top_n)
        if matched_name:
            if recs[0][3] < 20:
                st.warning("Sorry, we couldn't find close matches. Please try entering a different song.")
            else:
                df = pd.DataFrame(recs, columns=["Song", "Artist", "Genre", "Similarity (%)"])
                df.index += 1
                st.table(df)
                fig, ax = plt.subplots()
                ax.barh(df["Song"][::-1], df["Similarity (%)"][::-1], color="coral")
                ax.set_xlabel("Similarity (%)")
                st.pyplot(fig)
        else:
            st.warning("No close match found. Please check the song title.")

    st.button("⬅️ Back to Home", on_click=lambda: st.session_state.update({'page': None}))


# --------------------------------
# 🏠 HOME PAGE
def show_home():
    st.title(" Multi Recommendation System")
    st.markdown("Select a category below to get started:")

    col1, col2, col3 = st.columns([1,1,1])

    # ------------ Books ------------
    with col1:
        if st.button(" Books", use_container_width=True):
            st.session_state['page'] = 'Books'
        st.image("images.png", width=80)

    # ------------ Movies ------------
    with col2:
        if st.button(" Movies", use_container_width=True):
            st.session_state['page'] = 'Movies'
        st.image("movie_logo.png", width=80)

    # ------------ Music ------------
    with col3:
        if st.button(" Music", use_container_width=True):
            st.session_state['page'] = 'Music'
        st.image("Music_logo.png", width=80)

# --------------------------------
# 🚀 ROUTER
if st.session_state['page'] == 'Books':
    show_books()
elif st.session_state['page'] == 'Movies':
    show_movies()
elif st.session_state['page'] == 'Music':
    show_music()
else:
    show_home()
