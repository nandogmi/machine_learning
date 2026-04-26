import pandas as pd
from sklearn.model_selection import train_test_split

# =================================================================
# 1. CARGA DEL DATASET (Contexto NSL-KDD)
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
    df = pd.read_csv('KDDTrain+.txt', names=columns)
    print(f"Dataset original: {df.shape[0]} registros.")
except FileNotFoundError:
    print("Error: No se encontró el archivo de datos.")
    exit()

# Separamos características (X) de la etiqueta (y)
X = df.drop('label', axis=1)
y = df['label']

# =================================================================
# 2. DIVISIÓN SIMPLE (TRAIN / TEST)
# =================================================================
# Dividimos el 80% para entrenamiento y 20% para pruebas
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- División Simple (80/20) ---")
print(f"Entrenamiento: {len(X_train)} registros")
print(f"Pruebas:      {len(X_test)} registros")

# =================================================================
# 3. DIVISIÓN COMPLETA (TRAIN / VALIDATION / TEST)
# =================================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Del 80% restante, sacamos un 25% para Validación (0.25 * 0.8 = 0.2 del total)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

print("\n--- División Triple (60/20/20) ---")
print(f"Entrenamiento: {len(X_train)} registros (Ajuste de parámetros)")
print(f"Validación:    {len(X_val)} registros (Tuning del modelo)")
print(f"Pruebas:       {len(X_test)} registros (Evaluación final)")

# =================================================================
# 4. VERIFICACIÓN DE ESTRATIFICACIÓN
# =================================================================
print("\n--- Verificación de Proporciones (Stratify) ---")
print("Proporción en Original:\n", y.value_counts(normalize=True).head(3))
print("Proporción en Train:\n", y_train.value_counts(normalize=True).head(3))