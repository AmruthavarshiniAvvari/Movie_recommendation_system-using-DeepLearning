import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Deep Learning Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = tf.keras.models.load_model(
    "../models/movie_recommender.keras"
)

# ---------------------------------------------------
# LOAD ENCODERS
# ---------------------------------------------------

with open("../models/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("../models/movie_encoder.pkl", "rb") as f:
    movie_encoder = pickle.load(f)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

movies = pd.read_csv(
    "../data/u.item",
    sep="|",
    encoding="latin-1",
    header=None
)
genre_cols = [
    "unknown","Action","Adventure","Animation",
    "Children","Comedy","Crime","Documentary",
    "Drama","Fantasy","Film-Noir","Horror",
    "Musical","Mystery","Romance","Sci-Fi",
    "Thriller","War","Western"
]

movies.columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url"
] + genre_cols

genre_matrix = movies[genre_cols]

similarity_matrix = cosine_similarity(
    genre_matrix
)

ratings = pd.read_csv(
    "../data/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

num_movies = len(movie_encoder.classes_)

# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------

def recommend_movies(user_id, top_n=10):

    encoded_user = user_encoder.transform([user_id])[0]

    movie_ids = np.arange(num_movies)

    user_array = np.array(
        [encoded_user] * len(movie_ids)
    )

    predictions = model.predict(
        [user_array, movie_ids],
        verbose=0
    ).flatten()

    # convert back to 1-5 scale
    predictions = predictions * 5

    top_indices = predictions.argsort()[-top_n:][::-1]

    original_movie_ids = movie_encoder.inverse_transform(
        top_indices
    )

    recommendations = pd.DataFrame({
        "movie_id": original_movie_ids,
        "Predicted Rating":
        np.round(predictions[top_indices], 2)
    })

    recommendations = recommendations.merge(
        movies,
        on="movie_id"
    )

    recommendations.insert(
        0,
        "Rank",
        range(1, len(recommendations) + 1)
    )
    watched_movies = ratings[
        ratings["user_id"] == user_id
    ]["movie_id"].values

    movie_ids = np.array([
        movie
        for movie in range(num_movies)
        if movie_encoder.inverse_transform([movie])[0]
        not in watched_movies
    ])

    return recommendations[
        ["Rank", "title", "Predicted Rating"]
    ]

def get_similar_movies(movie_title, top_n=10):

    idx = movies[
        movies["title"] == movie_title
    ].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n+1]

    movie_indices = [
        i[0]
        for i in similarity_scores
    ]

    return movies.iloc[
        movie_indices
    ][["movie_id", "title"]]

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🎬 Hybrid Movie Recommendation System")

st.markdown("""
This application combines:

- Deep Learning Collaborative Filtering
- Content-Based Movie Similarity

to recommend movies using the MovieLens 100K dataset.
""")

# ---------------------------------------------------
# DATASET METRICS
# ---------------------------------------------------

st.sidebar.title("📊 Dataset Statistics")

st.sidebar.metric("Users", "943")
st.sidebar.metric("Movies", "1682")
st.sidebar.metric("Ratings", "100,000")
st.sidebar.metric("Test MAE", "0.81")

st.sidebar.markdown("---")

st.sidebar.info("""
MovieLens 100K Dataset

Deep Learning:
User + Movie Embeddings

Content-Based:
Genre Similarity
""")


tab1, tab2, tab3 = st.tabs(
    [
        "🎯 User Recommendations",
        "🎬 Similar Movies",
        "ℹ️ About"
    ]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("🎯 User-Based Recommendations")

    user_list = sorted(
        user_encoder.classes_
    )

    selected_user = st.selectbox(
        "Select User ID",
        user_list
    )

    user_ratings = ratings[
        ratings["user_id"] == selected_user
    ]

    st.info(
        f"""
        User ID: {selected_user}

        Movies Rated: {len(user_ratings)}

        Average Rating: {round(user_ratings['rating'].mean(),2)}
        """
    )

    if st.button("Recommend Movies", key="user_btn"):

        recommendations = recommend_movies(
            selected_user
        )

        st.success(
            f"Successfully generated recommendations for User {selected_user}"
        )

        st.subheader(
            "🏆 Top 10 Recommendations"
        )

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )

        chart_df = recommendations.sort_values(
            "Predicted Rating"
        )

        fig = px.bar(
            chart_df,
            x="Predicted Rating",
            y="title",
            orientation="h",
            title="Top Recommended Movies",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        csv = recommendations.to_csv(
            index=False
        )

        st.download_button(
            label="⬇ Download Recommendations",
            data=csv,
            file_name=f"user_{selected_user}_recommendations.csv",
            mime="text/csv"
        )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("🎬 Similar Movies")

    movie_title = st.selectbox(
        "Select Movie",
        sorted(movies["title"].unique())
    )

    if st.button(
        "Find Similar Movies",
        key="movie_btn"
    ):

        similar_movies = get_similar_movies(
            movie_title
        )

        st.success(
            f"Movies similar to: {movie_title}"
        )

        st.dataframe(
            similar_movies,
            use_container_width=True,
            hide_index=True
        )

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.subheader("ℹ️ About Project")

    st.markdown("""
    ### Dataset

    MovieLens 100K

    ### Deep Learning Model

    - User Embedding Layer
    - Movie Embedding Layer
    - Dense Neural Network

    ### Recommendation Techniques

    #### User-Based Recommendations

    Deep Learning Collaborative Filtering

    #### Movie-Based Recommendations

    Genre Similarity using Cosine Similarity

    ### Performance

    Test MAE ≈ 0.81

    ### Features

    ✅ User Recommendations

    ✅ Similar Movie Search

    ✅ Interactive Visualizations

    ✅ CSV Download
    """)

st.markdown("---")

st.markdown(
    "Built with ❤️ using TensorFlow, Streamlit and MovieLens 100K Dataset"
)


