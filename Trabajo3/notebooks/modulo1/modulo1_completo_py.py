"""
Módulo 1 - Predicción de Demanda (Placeholder para deployment)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_module1(horizon=30):
    """
    Función placeholder que simula predicciones de demanda
    """
    try:
        # Generar datos sintéticos para 5 rutas principales
        rutas = [
            "Bogotá - Medellín",
            "Bogotá - Cali", 
            "Medellín - Cartagena",
            "Cali - Barranquilla",
            "Bogotá - Bucaramanga"
        ]
        
        resultados = {}
        
        for ruta in rutas:
            # Generar demanda sintética
            np.random.seed(hash(ruta) % 1000)  # Seed consistente por ruta
            dates = [datetime.now() + timedelta(days=i) for i in range(horizon)]
            
            # Simular demanda con tendencia y ruido
            base_demand = 100 + np.random.normal(0, 10, horizon)
            trend = np.linspace(0, 20, horizon)
            seasonal = 15 * np.sin(np.linspace(0, 4*np.pi, horizon))
            demand = base_demand + trend + seasonal
            demand = np.maximum(demand, 0)  # Asegurar valores positivos
            
            # Crear DataFrame
            df = pd.DataFrame({
                'fecha': dates,
                'demanda': demand
            })
            
            # Crear gráfico de demanda
            fig_demand, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['fecha'], df['demanda'], marker='o', linewidth=2, markersize=4)
            ax.set_title(f'Predicción de Demanda - {ruta}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Fecha')
            ax.set_ylabel('Demanda Estimada')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Gráfico de descomposición simulado
            fig_decomp, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            # Serie original
            axes[0].plot(df['fecha'], df['demanda'])
            axes[0].set_title('Serie Original')
            axes[0].grid(True, alpha=0.3)
            
            # Tendencia
            axes[1].plot(df['fecha'], base_demand + trend)
            axes[1].set_title('Tendencia')
            axes[1].grid(True, alpha=0.3)
            
            # Estacionalidad
            axes[2].plot(df['fecha'], seasonal)
            axes[2].set_title('Componente Estacional')
            axes[2].grid(True, alpha=0.3)
            
            # Residuos
            residuals = np.random.normal(0, 5, horizon)
            axes[3].plot(df['fecha'], residuals)
            axes[3].set_title('Residuos')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            resultados[ruta] = {
                'fig_demand': fig_demand,
                'fig_decomp': fig_decomp,
                'datos': df
            }
        
        return resultados
        
    except Exception as e:
        print(f"Error en módulo 1: {e}")
        return {} 