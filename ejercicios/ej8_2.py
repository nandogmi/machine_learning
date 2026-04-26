import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler

# =================================================================
# FUNCIONES AUXILIARES
# =================================================================
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
# 1. LECTURA Y DIVISIÓN DEL DATASET (USANDO PANDAS NATIVO)
# =================================================================
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

try:
    # Reutilizamos el archivo de los ejercicios 6 y 7
    df = pd.read_csv('KDDTrain+.txt', names=columns)
    
    # El ARFF original no tiene la columna de dificultad, así que la borramos
    df = df.drop('difficulty', axis=1)
except FileNotFoundError:
    print("Error: No se encontró el archivo KDDTrain+.txt en el directorio.")
    exit()

train_set, val_set, test_set = train_val_test_split(df, stratify='protocol_type')

print(f"Longitud del Training Set: {len(train_set)}")
print(f"Longitud del Validation Set: {len(val_set)}")
print(f"Longitud del Test Set: {len(test_set)}")

# En el TXT la columna objetivo se llama 'label' en lugar de 'class'
X_train = train_set.drop("label", axis=1)
y_train = train_set["label"].copy()

# =================================================================
# 2. LIMPIEZA DE DATOS (Manejo de Valores Nulos)
# =================================================================
print("\n--- Limpieza de Datos ---")
X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"]<800), "src_bytes"] = np.nan
X_train.loc[(X_train["dst_bytes"]>500) & (X_train["dst_bytes"]<2000), "dst_bytes"] = np.nan

print("Aplicando SimpleImputer (Estrategia: Mediana)...")
X_train_num = X_train.select_dtypes(exclude=['object'])

imputer = SimpleImputer(strategy="median")
imputer.fit(X_train_num)
X_train_num_limpio = imputer.transform(X_train_num)

X_train_limpio = pd.DataFrame(X_train_num_limpio, columns=X_train_num.columns, index=X_train_num.index)
print("Valores nulos eliminados y sustituidos por la mediana.")

# =================================================================
# 3. TRANSFORMACIÓN DE VARIABLES CATEGÓRICAS
# =================================================================
print("\n--- Codificación Categórica ---")
protocol_type = X_train[['protocol_type']]

# 3.1 Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
protocol_encoded_ordinal = ordinal_encoder.fit_transform(protocol_type)
print("Categorías Ordinal detectadas:", ordinal_encoder.categories_)

# 3.2 One-Hot Encoding
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
protocol_encoded_oh = oh_encoder.fit_transform(protocol_type)
print(f"Dimensión de matriz One-Hot: {protocol_encoded_oh.shape}")

# 3.3 Get Dummies (Pandas)
df_dummies = pd.get_dummies(X_train['protocol_type']).head()
print("\nEjemplo de salida con Get Dummies (Primeros 5):\n", df_dummies)

# =================================================================
# 4. ESCALADO DE CARACTERÍSTICAS
# =================================================================
print("\n--- Escalado de Características ---")
scale_attrs = X_train_limpio[['src_bytes', 'dst_bytes']]

robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit_transform(scale_attrs)

df_scaled = pd.DataFrame(X_train_scaled, columns=['src_bytes_scaled', 'dst_bytes_scaled'], index=scale_attrs.index)
print("Datos escalados utilizando IQR (Primeros 5 registros):")
print(df_scaled.head())