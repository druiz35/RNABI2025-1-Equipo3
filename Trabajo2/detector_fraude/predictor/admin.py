from django.contrib import admin
from .models import Prediccion

@admin.register(Prediccion)
class PrediccionAdmin(admin.ModelAdmin):
    list_display = ('fecha_prediccion', 'estado', 'motivo_prestamo', 'score', 'probabilidad_incumplimiento')
    list_filter = ('estado', 'motivo_prestamo')
    search_fields = ('estado', 'motivo_prestamo', 'ocupacion')
    ordering = ('-fecha_prediccion',)
    readonly_fields = ('fecha_prediccion',)
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('estado', 'motivo_prestamo', 'ocupacion')
        }),
        ('Información Financiera', {
            'fields': ('ingresos_anuales', 'deuda_total', 'historial_credito')
        }),
        ('Resultados', {
            'fields': ('probabilidad_incumplimiento', 'score', 'fecha_prediccion')
        }),
    )
