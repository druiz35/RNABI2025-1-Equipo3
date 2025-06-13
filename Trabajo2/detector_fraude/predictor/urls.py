from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.home_intro, name='home_intro'),
    path('calculadora/', views.home, name='home'),
    path('resultado/<int:prediccion_id>/', views.resultado, name='resultado'),
    path('historial/', views.HistorialPredicciones.as_view(), name='historial'),
] 