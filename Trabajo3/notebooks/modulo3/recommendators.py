"""
Módulo 3 - Sistema de Recomendación (Placeholder para deployment)
"""
import pandas as pd
import numpy as np
import os

class CollaborativeRecommendator:
    """
    Recomendador colaborativo placeholder
    """
    
    def __init__(self):
        # Datos sintéticos de destinos
        self.destinations = [
            "Cartagena", "San Andrés", "Santa Marta", "Medellín", "Bogotá",
            "Cali", "Barranquilla", "Bucaramanga", "Pereira", "Armenia",
            "Manizales", "Ibagué", "Neiva", "Popayán", "Pasto"
        ]
        
        self.destination_types = [
            "Playa", "Isla", "Costa", "Ciudad", "Capital",
            "Valle", "Puerto", "Montaña", "Eje Cafetero", "Eje Cafetero",
            "Eje Cafetero", "Centro", "Sur", "Colonial", "Frontera"
        ]
        
        # Atributo requerido por app.py - corregido al archivo que existe
        self.MERGED_DF_PATH = "./notebooks/modulo3/m3_merged_df.csv"
    
    def recommend_destinations(self, user_id, top_n=5):
        """
        Genera recomendaciones sintéticas para un usuario
        """
        try:
            # Usar user_id como seed para consistencia
            np.random.seed(user_id % 1000)
            
            # Seleccionar destinos aleatorios
            indices = np.random.choice(len(self.destinations), size=top_n, replace=False)
            
            recommendations = []
            for i, idx in enumerate(indices):
                score = np.random.uniform(0.7, 0.95)  # Puntajes altos
                recommendations.append({
                    'destino': self.destinations[idx],
                    'tipo': self.destination_types[idx],
                    'score': score,
                    'ranking': i + 1
                })
            
            return pd.DataFrame(recommendations)
            
        except Exception as e:
            print(f"Error en recomendación: {e}")
            # Devolver recomendaciones por defecto
            default_recs = [
                {'destino': 'Cartagena', 'tipo': 'Playa', 'score': 0.95, 'ranking': 1},
                {'destino': 'Medellín', 'tipo': 'Ciudad', 'score': 0.92, 'ranking': 2},
                {'destino': 'San Andrés', 'tipo': 'Isla', 'score': 0.89, 'ranking': 3},
                {'destino': 'Santa Marta', 'tipo': 'Costa', 'score': 0.86, 'ranking': 4},
                {'destino': 'Bogotá', 'tipo': 'Capital', 'score': 0.83, 'ranking': 5}
            ]
            return pd.DataFrame(default_recs)
    
    def recommend(self, user_id, top_n=5):
        """
        Método requerido por app.py - wrapper de recommend_destinations
        """
        try:
            # Generar recomendaciones sintéticas
            np.random.seed(user_id % 1000)
            
            # Crear datos sintéticos con formato esperado por app.py
            recommendations = []
            for i in range(top_n):
                dest_idx = np.random.choice(len(self.destinations))
                recommendations.append({
                    'DestinationID': i + 1,
                    'Name': self.destinations[dest_idx],
                    'Category': self.destination_types[dest_idx],
                    'Rating': np.random.uniform(4.0, 5.0),
                    'PriceRange': np.random.choice(['$', '$$', '$$$']),
                    'Description': f"Hermoso destino en {self.destinations[dest_idx]}"
                })
            
            return pd.DataFrame(recommendations)
            
        except Exception as e:
            print(f"Error en recommend: {e}")
            # Datos por defecto
            default_data = [
                {'DestinationID': 1, 'Name': 'Cartagena', 'Category': 'Playa', 'Rating': 4.8, 'PriceRange': '$$', 'Description': 'Ciudad histórica y playas caribeñas'},
                {'DestinationID': 2, 'Name': 'Medellín', 'Category': 'Ciudad', 'Rating': 4.6, 'PriceRange': '$$', 'Description': 'Ciudad de la eterna primavera'},
                {'DestinationID': 3, 'Name': 'San Andrés', 'Category': 'Isla', 'Rating': 4.7, 'PriceRange': '$$$', 'Description': 'Paraíso caribeño'},
                {'DestinationID': 4, 'Name': 'Santa Marta', 'Category': 'Costa', 'Rating': 4.5, 'PriceRange': '$$', 'Description': 'Bahía y naturaleza exuberante'},
                {'DestinationID': 5, 'Name': 'Bogotá', 'Category': 'Capital', 'Rating': 4.3, 'PriceRange': '$$', 'Description': 'Capital cultural y gastronómica'}
            ]
            return pd.DataFrame(default_data)
    
    def get_user_names(self):
        """
        Devuelve lista de nombres de usuarios sintéticos
        """
        return [
            "Juan Pérez", "María García", "Carlos López", "Ana Rodríguez", "Luis Martín",
            "Laura Fernández", "Diego Silva", "Carmen Ruiz", "Miguel Torres", "Isabel Castro"
        ] 