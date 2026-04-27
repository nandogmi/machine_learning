import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# =================================================================
# 1. CARGA DE DATOS (NSL-KDD ARFF)
# =================================================================
def load_kdd_dataset_arff(data_path):
    try:
        data, meta = arff.loadarff(data_path)
        df = pd.DataFrame(data)
        # Decodificación de bytes a string para compatibilidad
        columnas_objeto = df.select_dtypes([object]).columns
        for col in columnas_objeto:
            df[col] = df[col].str.decode('utf-8')
        return df
    except Exception as e:
        print(f"Error al cargar ARFF: {e}")
        exit()

df = load_kdd_dataset_arff('KDDTrain+.arff')

# División inicial (60/20/20 como en el ejercicio anterior)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

# =================================================================
# 2. TRANSFORMADORES PERSONALIZADOS
# =================================================================

class EliminarColumnasIrrelevantes(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para limpiar columnas con varianza cero 
    o que no aportan información al modelo de red.
    """
    def __init__(self, columnas_a_eliminar=None):
        self.columnas_a_eliminar = columnas_a_eliminar
    
    def fit(self, X, y=None):
        return self # No necesita aprender nada de los datos
    
    def transform(self, X):
        return X.drop(self.columnas_a_eliminar, axis=1, errors='ignore')

# =================================================================
# 3. CONSTRUCCIÓN DEL PIPELINE
# =================================================================

# 3.1 Identificación de columnas por tipo
lista_num = X_train.select_dtypes(exclude=['object']).columns.tolist()
lista_cat = X_train.select_dtypes(include=['object']).columns.tolist()

# 3.2 Pipeline para atributos numéricos
# Flujo: Imputación (mediana) -> Escalado Robusto
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('robust_scaler', RobustScaler())
])

# 3.3 Combinador Global (ColumnTransformer)
# Aplica transformaciones específicas a cada tipo de columna en un solo paso.
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, lista_num),
    ("cat", OneHotEncoder(handle_unknown='ignore'), lista_cat),
    ("eliminar", EliminarColumnasIrrelevantes(columnas_a_eliminar=['num_outbound_cmds']), lista_num + lista_cat)
], remainder='passthrough')

# =================================================================
# 4. EJECUCIÓN DEL PIPELINE
# =================================================================
print("\nEjecutando Pipeline completo...")
# Aquí es donde tu i7-11800H procesa todo el flujo de forma optimizada
X_train_preparado = full_pipeline.fit_transform(X_train)

print(f"Forma de los datos originales: {X_train.shape}")
print(f"Forma de los datos preparados: {X_train_preparado.shape}")
print("\nEl Pipeline ha procesado, imputado, escalado y codificado todo el dataset.")