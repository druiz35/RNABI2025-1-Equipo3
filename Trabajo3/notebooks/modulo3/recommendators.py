
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import numpy as np
import pandas as pd
import kagglehub
import os

# EJEMPLO DE USO
"""
recomendador = CollaborativeRecommendator()

# Para hacer una recomendación. Devuelve un dataframe de pandas
user_id = 3
recommendator.recommend(user_id)
"""

class CollaborativeRecommendator:
    MERGED_DF_PATH = "./notebooks/modulo3/m3_merged_df.csv."  # Ruta local del DataFrame fusionado con toda la información
    USERHISTORY_DF_PATH = "./"              # (No se usa explícitamente, puedes eliminarla o usarla para futuras rutas)

    def __init__(self):
        self.merged_df = None
        self.user_similarity = None
        self.load_dfs()
        self.setup()  
    
    def load_dfs(self):
        # Descarga y carga datasets desde Kaggle usando kagglehub
        general_path = kagglehub.dataset_download("amanmehra23/travel-recommendation-dataset")

        destinations_path = general_path + "/Expanded_Destinations.csv"
        self.destinations_df = pd.read_csv(destinations_path)

        reviews_path = general_path + "/Final_Updated_Expanded_Reviews.csv"
        self.reviews_df = pd.read_csv(reviews_path)

        userhistory_path = general_path + "/Final_Updated_Expanded_UserHistory.csv"
        self.userhistory_df = pd.read_csv(userhistory_path)

        users_path = general_path + "/Final_Updated_Expanded_Users.csv"
        self.users_df = pd.read_csv(users_path)

        # Carga el DataFrame fusionado desde CSV local
        
        print("-")
        print("-")
        print("-")

        print(os.getcwd())

        print("-")
        print("-")
        print("-")

        df = pd.read_csv(CollaborativeRecommendator.MERGED_DF_PATH)

        # Elimina columnas innecesarias (las dos primeras)
        df.drop(df.columns[[0,1]], axis=1, inplace=True)

        # Crea una columna descriptiva combinando varias columnas en texto
        # Nota: Hay que convertir a string para evitar errores (revisar si es necesario)
        df["description"] = (
            df['Type'].astype(str) + ' ' +
            df['State'].astype(str) + ' ' +
            df['BestTimeToVisit'].astype(str) + ' ' +
            df['Preferences'].astype(str) + ' ' +
            df['Gender'].astype(str) + ' ' +
            df['NumberOfAdults'].astype(str) + ' ' +
            df['NumberOfChildren'].astype(str)
        )

        self.merged_df = df

    def setup(self):
        # Puedes descomentar para usar vectorización basada en contenido (por ejemplo)
        # vectorizer = CountVectorizer(stop_words='english')
        # vectorizer = TfidfVectorizer(stop_words='english')
        # destination_features = vectorizer.fit_transform(self.merged_df['description'])
        # cosine_sim = cosine_similarity(destination_features, destination_features)

        # Construimos matriz usuario x destino con ratings
        user_item_matrix = self.userhistory_df.pivot(
            index='UserID',
            columns='DestinationID',
            values='ExperienceRating'
        )
        # Rellenamos NaN con 0 (no calificado)
        self.user_item_matrix = user_item_matrix.fillna(0)

        # Calculamos similitud coseno entre usuarios para recomendación colaborativa
        self.user_similarity = cosine_similarity(self.user_item_matrix)

    def recommend(self, user_id):
        # Obtenemos el vector de similitud del usuario objetivo contra todos los usuarios
        similar_users = self.user_similarity[user_id - 1]

        # Ordenamos los usuarios más similares, ignorando el mismo usuario (índice 0)
        similar_users_idx = np.argsort(similar_users)[::-1][1:6]

        # Promediamos las calificaciones de esos usuarios similares
        similar_user_ratings = self.user_item_matrix.iloc[similar_users_idx].mean(axis=0)

        # Seleccionamos los 5 destinos con mejores puntuaciones promedio
        recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(5).index

        # Filtramos la info de destinos para las recomendaciones
        recommendations = self.destinations_df[
            self.destinations_df['DestinationID'].isin(recommended_destinations_ids)
        ][[
            'DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit'
        ]]
        return recommendations
