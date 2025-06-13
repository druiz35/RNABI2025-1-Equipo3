from django import forms

# Opciones categóricas según el análisis y los datos del notebook
TERM_CHOICES = [
    (' 36 months', '36 months'),
    (' 60 months', '60 months'),
]

# 35 sub_grades
SUB_GRADE_CHOICES = [
    ('A1', 'A1'), ('A2', 'A2'), ('A3', 'A3'), ('A4', 'A4'), ('A5', 'A5'),
    ('B1', 'B1'), ('B2', 'B2'), ('B3', 'B3'), ('B4', 'B4'), ('B5', 'B5'),
    ('C1', 'C1'), ('C2', 'C2'), ('C3', 'C3'), ('C4', 'C4'), ('C5', 'C5'),
    ('D1', 'D1'), ('D2', 'D2'), ('D3', 'D3'), ('D4', 'D4'), ('D5', 'D5'),
    ('E1', 'E1'), ('E2', 'E2'), ('E3', 'E3'), ('E4', 'E4'), ('E5', 'E5'),
    ('F1', 'F1'), ('F2', 'F2'), ('F3', 'F3'), ('F4', 'F4'), ('F5', 'F5'),
    ('G1', 'G1'), ('G2', 'G2'), ('G3', 'G3'), ('G4', 'G4'), ('G5', 'G5'),
]

EMP_LENGTH_CHOICES = [
    ('< 1 year', '< 1 year'),
    ('1 year', '1 year'),
    ('2 years', '2 years'),
    ('3 years', '3 years'),
    ('4 years', '4 years'),
    ('5 years', '5 years'),
    ('6 years', '6 years'),
    ('7 years', '7 years'),
    ('8 years', '8 years'),
    ('9 years', '9 years'),
    ('10+ years', '10+ years'),
    ('nan', 'No especificado'),
]

HOME_OWNERSHIP_CHOICES = [
    ('RENT', 'RENT'),
    ('OWN', 'OWN'),
    ('MORTGAGE', 'MORTGAGE'),
]

VERIFICATION_STATUS_CHOICES = [
    ('Verified', 'Verified'),
    ('Source Verified', 'Source Verified'),
    ('Not Verified', 'Not Verified'),
]

PURPOSE_CHOICES = [
    ('credit_card', 'Credit Card'),
    ('car', 'Car'),
    ('small_business', 'Small Business'),
    ('other', 'Other'),
    ('wedding', 'Wedding'),
    ('debt_consolidation', 'Debt Consolidation'),
    ('home_improvement', 'Home Improvement'),
    ('major_purchase', 'Major Purchase'),
    ('medical', 'Medical'),
    ('moving', 'Moving'),
    ('vacation', 'Vacation'),
    ('house', 'House'),
    ('renewable_energy', 'Renewable Energy'),
    ('educational', 'Educational'),
]

# 51 US states
ADDR_STATE_CHOICES = [
    ('AL', 'Alabama'), ('AK', 'Alaska'), ('AZ', 'Arizona'), ('AR', 'Arkansas'),
    ('CA', 'California'), ('CO', 'Colorado'), ('CT', 'Connecticut'), ('DE', 'Delaware'),
    ('FL', 'Florida'), ('GA', 'Georgia'), ('HI', 'Hawaii'), ('ID', 'Idaho'),
    ('IL', 'Illinois'), ('IN', 'Indiana'), ('IA', 'Iowa'), ('KS', 'Kansas'),
    ('KY', 'Kentucky'), ('LA', 'Louisiana'), ('ME', 'Maine'), ('MD', 'Maryland'),
    ('MA', 'Massachusetts'), ('MI', 'Michigan'), ('MN', 'Minnesota'), ('MS', 'Mississippi'),
    ('MO', 'Missouri'), ('MT', 'Montana'), ('NE', 'Nebraska'), ('NV', 'Nevada'),
    ('NH', 'New Hampshire'), ('NJ', 'New Jersey'), ('NM', 'New Mexico'), ('NY', 'New York'),
    ('NC', 'North Carolina'), ('ND', 'North Dakota'), ('OH', 'Ohio'), ('OK', 'Oklahoma'),
    ('OR', 'Oregon'), ('PA', 'Pennsylvania'), ('RI', 'Rhode Island'), ('SC', 'South Carolina'),
    ('SD', 'South Dakota'), ('TN', 'Tennessee'), ('TX', 'Texas'), ('UT', 'Utah'),
    ('VT', 'Vermont'), ('VA', 'Virginia'), ('WA', 'Washington'), ('WV', 'West Virginia'),
    ('WI', 'Wisconsin'), ('WY', 'Wyoming'), ('DC', 'District of Columbia'),
]

INITIAL_LIST_STATUS_CHOICES = [
    ('f', 'f'),
    ('w', 'w'),
]

class PrediccionForm(forms.Form):
    # Categóricas
    term = forms.ChoiceField(choices=TERM_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    sub_grade = forms.ChoiceField(choices=SUB_GRADE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    emp_length = forms.ChoiceField(choices=EMP_LENGTH_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    home_ownership = forms.ChoiceField(choices=HOME_OWNERSHIP_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    verification_status = forms.ChoiceField(choices=VERIFICATION_STATUS_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    purpose = forms.ChoiceField(choices=PURPOSE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    addr_state = forms.ChoiceField(choices=ADDR_STATE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    initial_list_status = forms.ChoiceField(choices=INITIAL_LIST_STATUS_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))

    # Numéricas
    loan_amnt = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    funded_amnt = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    funded_amnt_inv = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    int_rate = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    installment = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    annual_inc = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    loan_status = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    dti = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    delinq_2yrs = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    inq_last_6mths = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    mths_since_last_delinq = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    open_acc = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    pub_rec = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    revol_bal = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    revol_util = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    total_acc = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    out_prncp = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    out_prncp_inv = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    collections_12_mths_ex_med = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    mths_since_last_major_derog = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    acc_now_delinq = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    tot_coll_amt = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    tot_cur_bal = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    total_rev_hi_lim = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'})) 