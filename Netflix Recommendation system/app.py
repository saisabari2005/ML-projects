import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import bs4 as bs

# Load the NLP model and TF-IDF vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

def create_similarity():
    data = pd.read_csv('final_data (1).csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return 'Sorry! Try another movie name'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        l = [data['movie_title'][a] for a, _ in lst]
        return l

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    data = pd.read_csv('final_data (1).csv')
    return list(data['movie_title'].str.capitalize())

def get_movie_details(title, poster, overview, vote_average, vote_count, genres, release_date, runtime, status, imdb_id):
    return {
        "title": title,
        "poster": poster,
        "overview": overview,
        "vote_average": vote_average,
        "vote_count": vote_count,
        "genres": genres,
        "release_date": release_date,
        "runtime": runtime,
        "status": status,
        "imdb_id": imdb_id
    }

def get_cast_details(cast_names, cast_profiles, cast_bdays, cast_places, cast_bios, cast_ids):
    return {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

def get_reviews(imdb_id):
    sauce = urllib.request.urlopen(f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt').read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})
    
    reviews_list = []
    reviews_status = []
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')
    
    return {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

def get_recommended_movies(rec_movies, rec_posters):
    return {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

st.title('Netflix Recommender System')

# User selects a movie
selected_movie = st.selectbox('Type or select a movie from the dropdown', get_suggestions())

if st.button('Show Recommendations'):
    recommendations = rcmd(selected_movie)
    if type(recommendations) == str:
        st.write(recommendations)
    else:
        st.write('Recommended Movies:')
        for movie in recommendations:
            st.write(movie)
    
    # Example data to replace the AJAX request
    title = "Example Movie"
    cast_ids = "[1, 2]"
    cast_names = '["Actor 1", "Actor 2"]'
    cast_chars = '["Character 1", "Character 2"]'
    cast_bdays = '["1980-01-01", "1990-01-01"]'
    cast_bios = '["Biography 1", "Biography 2"]'
    cast_places = '["Place 1", "Place 2"]'
    cast_profiles = '["https://example.com/actor1.jpg", "https://example.com/actor2.jpg"]'
    imdb_id = "tt1234567"
    poster = "https://example.com/poster.jpg"
    genres = "Drama, Action"
    overview = "This is a brief overview of the movie."
    vote_average = 8.0
    vote_count = 1234
    release_date = "2022-01-01"
    runtime = "120 minutes"
    status = "Released"
    rec_movies = '["Movie 1", "Movie 2"]'
    rec_posters = '["https://example.com/poster1.jpg", "https://example.com/poster2.jpg"]'

    # Process the data
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # Convert cast_ids to list
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # Render the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')
    
    # Combine multiple lists as a dictionary
    movie_cards = get_recommended_movies(rec_movies, rec_posters)
    cast_details = get_cast_details(cast_names, cast_profiles, cast_bdays, cast_places, cast_bios, cast_ids)

    # Fetch movie details
    movie_details = get_movie_details(title, poster, overview, vote_average, vote_count, genres, release_date, runtime, status, imdb_id)
    
    # Display movie details
    st.image(movie_details['poster'], caption=movie_details['title'], use_column_width=True)
    st.write(f"**Main Title:** {movie_details['title']}")
    st.write(f"**Overview:** {movie_details['overview']}")
    st.write(f"**Rating:** {movie_details['vote_average']}/10 ({movie_details['vote_count']} votes)")
    st.write(f"**Genre:** {movie_details['genres']}")
    st.write(f"**Release Date:** {movie_details['release_date']}")
    st.write(f"**Runtime:** {movie_details['runtime']}")
    st.write(f"**Status:** {movie_details['status']}")

    # Display cast details
    st.write("## Actors & Actresses")
    for name, details in cast_details.items():
        with st.expander(name):
            st.image(details[1], width=150)
            st.write(f"**Birthday:** {details[2]}")
            st.write(f"**Place of Birth:** {details[3]}")
            st.write(f"**Biography:** {details[4]}")

    # Display user reviews
    st.write("## User Reviews")
    reviews = get_reviews(imdb_id)
    for review, status in reviews.items():
        st.write(f"**Review:** {review}")
        st.write(f"**Status:** {status}")
        st.write("😊" if status == 'Good' else "😞")

    # Display recommended movies
    st.write("## Recommended Movies For You")
    for poster, title in movie_cards.items():
        st.image(poster, width=150)
        st.write(f"**Title:** {title}")
