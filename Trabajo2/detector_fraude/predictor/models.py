from django.db import models

# Create your models here.

class Prediccion(models.Model):
    fecha_prediccion = models.DateTimeField(auto_now_add=True)
    estado = models.CharField(max_length=50)
    motivo_prestamo = models.CharField(max_length=100)
    ocupacion = models.CharField(max_length=100)
    probabilidad_incumplimiento = models.FloatField()
    score = models.IntegerField()
    
    # Campos adicionales que se agregarán según el modelo final
    # Por ejemplo:
    ingresos_anuales = models.FloatField(null=True, blank=True)
    deuda_total = models.FloatField(null=True, blank=True)
    historial_credito = models.CharField(max_length=20, null=True, blank=True)
    
    class Meta:
        verbose_name = 'Predicción'
        verbose_name_plural = 'Predicciones'
        ordering = ['-fecha_prediccion']
    
    def __str__(self):
        return f"Predicción {self.id} - {self.fecha_prediccion.strftime('%Y-%m-%d %H:%M')}"
