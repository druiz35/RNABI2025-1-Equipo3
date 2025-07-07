"""
Módulo 3 - Sistema de Recomendación (Placeholder para deployment)
"""
import pandas as pd
import numpy as np

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
    
    def get_user_names(self):
        """
        Devuelve lista de nombres de usuarios sintéticos
        """
        return [
            "Juan Pérez", "María García", "Carlos López", "Ana Rodríguez", "Luis Martín",
            "Laura Fernández", "Diego Silva", "Carmen Ruiz", "Miguel Torres", "Isabel Castro"
        ] 