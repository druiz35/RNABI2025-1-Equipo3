
from sklearn.model_selection import train_test_split as sklearn_train_test_split


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import numpy as np
import pandas as pd

import kagglehub


# EJEMPLO DE USO
"""
recomendador = CollaborativeRecommendator()

# Para hacer una recomendación. Devuelve un dataframe de pandas
user_id = 3
recommendator.recommend(user_id)
"""

class CollaborativeRecommendator:
    MERGED_DF_PATH = "./m3_merged_df.csv"
    USERHISTORY_DF_PATH = "./"
    def __init__(self):
        self.merged_df = None
        self.user_similarity = None
        self.load_dfs()
        self.setup()  
    
    def load_dfs(self):
        general_path = kagglehub.dataset_download("amanmehra23/travel-recommendation-dataset")
        destinations_path = general_path + "/Expanded_Destinations.csv"
        self.destinations_df = pd.read_csv(destinations_path)

        reviews_path = general_path + "/Final_Updated_Expanded_Reviews.csv"
        self.reviews_df = pd.read_csv(reviews_path)

        userhistory_path = general_path + "/Final_Updated_Expanded_UserHistory.csv"
        self.userhistory_df = pd.read_csv(userhistory_path)

        users_path = general_path + "/Final_Updated_Expanded_Users.csv"
        self.users_df = pd.read_csv(users_path)

        df = pd.read_csv(CollaborativeRecommendator.MERGED_DF_PATH)
        df.drop(df.columns[[0,1]], axis=1)
        df["description"]= f"{df['Type']} {df['State']} {df['BestTimeToVisit']} {df['Preferences']} {df["Gender"]} {df["NumberOfAdults"]} {df["NumberOfChildren"]}"
        self.merged_df = df


    def setup(self):
        #vectorizer = CountVectorizer(stop_words='english')
        #vectorizer = TfidfVectorizer(stop_words='english')
        #destination_features = vectorizer.fit_transform(self.merged_df['description'])
        #cosine_sim = cosine_similarity(destination_features, destination_features)
        user_item_matrix = self.userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')
        self.user_item_matrix = user_item_matrix.fillna(0)
        self.user_similarity = cosine_similarity(user_item_matrix)

    def recommend(self, user_id):
        similar_users = self.user_similarity[user_id - 1]
        similar_users_idx = np.argsort(similar_users)[::-1][1:6]
        similar_user_ratings = self.user_item_matrix.iloc[similar_users_idx].mean(axis=0)
        recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(5).index
        recommendations = self.destinations_df[self.destinations_df['DestinationID'].isin(recommended_destinations_ids)][[
            'DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit'
        ]]
        return recommendations
