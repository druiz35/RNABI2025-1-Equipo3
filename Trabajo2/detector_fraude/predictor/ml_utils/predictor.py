import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from django.conf import settings

class MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def get_model_path():
    """Retorna la ruta al archivo del modelo"""
    return os.path.join(settings.BASE_DIR, 'modelo', 'mlp_model.pt')

def load_model():
    """Carga el modelo desde el archivo .pt"""
    input_dim = 63  # Número de características según el notebook
    n_classes = 2   # Número de clases (0: paga, 1: no paga)
    
    model = MLP(input_dim, n_classes)
    model_path = get_model_path()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Poner el modelo en modo evaluación
        return model
    else:
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

def prepare_input_data(data):
    """
    Prepara los datos de entrada para el modelo.
    Args:
        data (dict): Diccionario con los datos del formulario
    Returns:
        torch.Tensor: Tensor con los datos preparados para el modelo
    """
    # Columnas esperadas por el modelo
    columns = [
        'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment',
        'sub_grade', 'emp_length', 'annual_inc', 'loan_status', 'dti', 'delinq_2yrs',
        'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
        'total_acc', 'out_prncp', 'out_prncp_inv', 'collections_12_mths_ex_med',
        'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
        # Categóricas one-hot:
        'term_ 36 months', 'term_ 60 months',
        'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT',
        'verification_status_Not Verified', 'verification_status_Source Verified', 'verification_status_Verified',
        'initial_list_status_f', 'initial_list_status_w',
        'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational',
        'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical',
        'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
        'purpose_vacation', 'purpose_wedding',
        # Estado hash (8 columnas)
        'addr_state_hash_0', 'addr_state_hash_1', 'addr_state_hash_2', 'addr_state_hash_3',
        'addr_state_hash_4', 'addr_state_hash_5', 'addr_state_hash_6', 'addr_state_hash_7',
    ]
    input_data = pd.DataFrame(0, index=[0], columns=columns)

    # Numéricas directas
    for col in [
        'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment',
        'annual_inc', 'loan_status', 'dti', 'delinq_2yrs', 'inq_last_6mths',
        'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
        'total_acc', 'out_prncp', 'out_prncp_inv', 'collections_12_mths_ex_med',
        'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']:
        input_data.at[0, col] = float(data.get(col, 0))

    # sub_grade: mapear a número (A1=1, ..., G5=35)
    sub_grade_map = {f"{l}{n}": i+1 for i, (l, n) in enumerate([(l, n) for l in 'ABCDEFG' for n in range(1,6)])}
    input_data.at[0, 'sub_grade'] = sub_grade_map.get(data.get('sub_grade'), 0)

    # emp_length: mapear a número (pero el modelo espera string, así que lo dejamos como string si es necesario)
    emp_length_map = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10,
        'nan': 0
    }
    input_data.at[0, 'emp_length'] = emp_length_map.get(data.get('emp_length'), 0)

    # term (one-hot)
    term = data.get('term')
    if term:
        input_data.at[0, f'term_{term}'] = 1

    # home_ownership (one-hot)
    ho = data.get('home_ownership')
    if ho:
        input_data.at[0, f'home_ownership_{ho}'] = 1

    # verification_status (one-hot)
    vs = data.get('verification_status')
    if vs:
        input_data.at[0, f'verification_status_{vs}'] = 1

    # initial_list_status (one-hot)
    ils = data.get('initial_list_status')
    if ils:
        input_data.at[0, f'initial_list_status_{ils}'] = 1

    # purpose (one-hot)
    purpose = data.get('purpose')
    if purpose:
        input_data.at[0, f'purpose_{purpose}'] = 1

    # addr_state (hash one-hot)
    addr_state = data.get('addr_state')
    if addr_state:
        estado_hash = hash(addr_state) % 8
        input_data.at[0, f'addr_state_hash_{estado_hash}'] = 1

    # Convertir a tensor de PyTorch
    input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
    return input_tensor

def predict(data):
    """
    Realiza una predicción con el modelo.
    
    Args:
        data (dict): Diccionario con los datos del formulario
        
    Returns:
        tuple: (probabilidad_incumplimiento, score)
    """
    try:
        model = load_model()
        input_tensor = prepare_input_data(data)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Probabilidad de incumplimiento (clase 1)
            prob_incumplimiento = probabilities[0, 1].item() * 100
            
            # Calcular score (850 - probabilidad_incumplimiento * 5)
            # Esto da un rango aproximado de 350-850 donde mayor score es mejor
            score = int(850 - prob_incumplimiento * 5)
            score = max(350, min(850, score))  # Limitar entre 350 y 850
            
            return prob_incumplimiento, score
    except Exception as e:
        print(f"Error al realizar la predicción: {str(e)}")
        # Valores por defecto en caso de error
        return 25.0, 750 