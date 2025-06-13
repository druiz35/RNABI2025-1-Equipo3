from django.shortcuts import render, redirect
from django.views.generic import ListView
from django.contrib import messages
from .forms import PrediccionForm
from .models import Prediccion
from .ml_utils.predictor import predict
import os

def home_intro(request):
    return render(request, 'predictor/home_intro.html')

def home(request):
    if request.method == 'POST':
        form = PrediccionForm(request.POST)
        if form.is_valid():
            # Tomar todos los datos del formulario dinámicamente
            datos = form.cleaned_data.copy()
            # Realizar la predicción
            try:
                probabilidad, score = predict(datos)
            except Exception as e:
                messages.error(request, f'Error al realizar la predicción: {str(e)}')
                return render(request, 'predictor/home.html', {'form': form})
            # Guardar la predicción con los campos relevantes
            prediccion = Prediccion.objects.create(
                probabilidad_incumplimiento=probabilidad,
                score=score,
                ingresos_anuales=datos.get('annual_inc'),
                deuda_total=datos.get('tot_cur_bal'),
                historial_credito=datos.get('emp_length'),
                estado=datos.get('addr_state'),
                motivo_prestamo=datos.get('purpose'),
                ocupacion='-',
            )
            messages.success(request, 'Predicción realizada con éxito')
            return redirect('predictor:resultado', prediccion_id=prediccion.id)
    else:
        form = PrediccionForm()
    return render(request, 'predictor/home.html', {'form': form})

def resultado(request, prediccion_id):
    prediccion = Prediccion.objects.get(id=prediccion_id)
    return render(request, 'predictor/resultado.html', {'prediccion': prediccion})

class HistorialPredicciones(ListView):
    model = Prediccion
    template_name = 'predictor/historial.html'
    context_object_name = 'predicciones'
    paginate_by = 10
