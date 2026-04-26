import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
# FUNCIÓN AUXILIAR DE VISUALIZACIÓN
# =================================================================
def graficar_estratificacion(datasets, etiquetas, titulo):
    """
    Genera un gráfico de barras comparando la proporción de las 5 clases
    más frecuentes en distintos subconjuntos de datos.
    """
    plt.figure(figsize=(12, 6))
    
    # Extraemos el top 5 de clases del dataset original para no saturar el gráfico
    top_clases = datasets[0].value_counts().head(5).index
    
    df_plot = pd.DataFrame()
    for serie, etiqueta in zip(datasets, etiquetas):
        # Calcular proporciones normalizadas (porcentajes)
        proporciones = serie.value_counts(normalize=True) * 100
        
        # Filtrar solo las clases principales y llenar con 0 si alguna no estuviera
        proporciones = proporciones.reindex(top_clases).fillna(0)
        
        df_temp = pd.DataFrame({
            'Clase': proporciones.index,
            'Porcentaje (%)': proporciones.values,
            'Conjunto': etiqueta
        })
        df_plot = pd.concat([df_plot, df_temp])
        
    sns.barplot(data=df_plot, x='Clase', y='Porcentaje (%)', hue='Conjunto', palette='tab10')
    plt.title(titulo)
    plt.ylabel('Proporción en el conjunto (%)')
    plt.xlabel('Tipos de Ataque / Tráfico')
    plt.tight_layout()
    plt.show()

# =================================================================
# 2. DIVISIÓN SIMPLE (TRAIN / TEST)
# =================================================================
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- División Simple (80/20) ---")
print(f"Entrenamiento: {len(X_train_simple)} registros")
print(f"Pruebas:      {len(X_test_simple)} registros")

# Verificación Visual 1
graficar_estratificacion(
    datasets=[y, y_train_simple, y_test_simple],
    etiquetas=['Original (100%)', 'Train (80%)', 'Test (20%)'],
    titulo='Verificación de Estratificación: División Simple'
)

# =================================================================
# 3. DIVISIÓN COMPLETA (TRAIN / VALIDATION / TEST)
# =================================================================
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_triple, X_val_triple, y_train_triple, y_val_triple = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

print("\n--- División Triple (60/20/20) ---")
print(f"Entrenamiento: {len(X_train_triple)} registros")
print(f"Validación:    {len(X_val_triple)} registros")
print(f"Pruebas:       {len(X_test_full)} registros")

# Verificación Visual 2
graficar_estratificacion(
    datasets=[y, y_train_triple, y_val_triple, y_test_full],
    etiquetas=['Original (100%)', 'Train (60%)', 'Validation (20%)', 'Test (20%)'],
    titulo='Verificación de Estratificación: División Triple'
)