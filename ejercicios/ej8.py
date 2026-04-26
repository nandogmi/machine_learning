import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler

# =================================================================
# FUNCIONES AUXILIARES
# =================================================================
def load_kdd_dataset(data_path):
    """Lectura del conjunto de datos NSL-KDD en formato ARFF."""
    try:
        with open(data_path, 'r') as train_set:
            dataset = arff.load(train_set)
        attributes = [attr[0] for attr in dataset["attributes"]]
        return pd.DataFrame(dataset["data"], columns=attributes)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {data_path}")
        exit()

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """Divide el dataset en Train (60%), Validation (20%) y Test (20%)."""
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    return train_set, val_set, test_set

# =================================================================
# 1. LECTURA Y DIVISIÓN DEL DATASET
# =================================================================
# NOTA: Ajusta la ruta del archivo según tu estructura de directorios
archivo_arff = "KDDTrain+.arff" 
df = load_kdd_dataset(archivo_arff)

train_set, val_set, test_set = train_val_test_split(df, stratify='protocol_type')

print(f"Longitud del Training Set: {len(train_set)}")
print(f"Longitud del Validation Set: {len(val_set)}")
print(f"Longitud del Test Set: {len(test_set)}")

# Separar características (X) de las etiquetas (y)
X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

# =================================================================
# 2. LIMPIEZA DE DATOS (Manejo de Valores Nulos)
# =================================================================
print("\n--- Limpieza de Datos ---")
# Simulación de pérdida de datos (para propósitos educativos)
X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"]<800), "src_bytes"] = np.nan
X_train.loc[(X_train["dst_bytes"]>500) & (X_train["dst_bytes"]<2000), "dst_bytes"] = np.nan

# Método de imputación utilizando la Mediana vía scikit-learn
print("Aplicando SimpleImputer (Estrategia: Mediana)...")
# El imputer no admite texto, separamos las columnas numéricas
X_train_num = X_train.select_dtypes(exclude=['object'])

imputer = SimpleImputer(strategy="median")
imputer.fit(X_train_num)
X_train_num_limpio = imputer.transform(X_train_num)

# Reconstruir el DataFrame numérico sin valores nulos
X_train_limpio = pd.DataFrame(X_train_num_limpio, columns=X_train_num.columns, index=X_train_num.index)
print("Valores nulos eliminados exitosamente.")

# =================================================================
# 3. TRANSFORMACIÓN DE VARIABLES CATEGÓRICAS
# =================================================================
print("\n--- Codificación Categórica ---")
protocol_type = X_train[['protocol_type']]

# 3.1 Ordinal Encoding (Asigna un número entero a cada categoría)
ordinal_encoder = OrdinalEncoder()
protocol_encoded_ordinal = ordinal_encoder.fit_transform(protocol_type)
print("Categorías Ordinal:", ordinal_encoder.categories_)

# 3.2 One-Hot Encoding (Genera una matriz binaria)
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
protocol_encoded_oh = oh_encoder.fit_transform(protocol_type)
print(f"Dimensión de matriz One-Hot: {protocol_encoded_oh.shape}")

# 3.3 Get Dummies (Alternativa nativa de Pandas)
df_dummies = pd.get_dummies(X_train['protocol_type']).head()
print("Ejemplo Get Dummies (Pandas):\n", df_dummies)

# =================================================================
# 4. ESCALADO DE CARACTERÍSTICAS
# =================================================================
print("\n--- Escalado de Características ---")
# Aplicamos RobustScaler a dos métricas de tráfico
scale_attrs = X_train_limpio[['src_bytes', 'dst_bytes']]

robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit_transform(scale_attrs)

# Mostrar el resultado final
df_scaled = pd.DataFrame(X_train_scaled, columns=['src_bytes_scaled', 'dst_bytes_scaled'], index=scale_attrs.index)
print("Datos escalados (Primeros 5 registros):")
print(df_scaled.head())