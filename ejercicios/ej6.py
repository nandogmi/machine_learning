import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# =================================================================
# Definición completa de las 43 columnas del dataset NSL-KDD
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
    # Cargamos el dataset (ajusta 'KDDTrain+.txt' al nombre de tu archivo)
    df = pd.read_csv('KDDTrain+.txt', names=columns)
    print(f"Dataset cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
except FileNotFoundError:
    print("Error: No se encontró el archivo KDDTrain+.txt en el directorio.")
    exit()

# =================================================================
# 2. VISUALIZACIONES COMPLETAS DEL CUADERNO
# =================================================================

# --- VISUALIZACIÓN 1: Distribución de la Clase Objetivo (Label) ---
plt.figure(figsize=(12, 6))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Distribución de Tipos de Tráfico / Ataques')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# --- VISUALIZACIÓN 2: Protocolos de Red ---
plt.figure(figsize=(8, 8))
df['protocol_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Proporción de Protocolos (TCP, UDP, ICMP)')
plt.ylabel('')
plt.show()

# --- VISUALIZACIÓN 3: Top 10 Servicios de Red más Frecuentes ---
plt.figure(figsize=(12, 6))
top_services = df['service'].value_counts().head(10)
sns.barplot(x=top_services.index, y=top_services.values, palette='magma')
plt.title('Top 10 Servicios con Mayor Tráfico')
plt.xlabel('Servicio')
plt.ylabel('Cantidad de Conexiones')
plt.show()

# --- VISUALIZACIÓN 4: Distribución de Flags de Conexión ---
plt.figure(figsize=(10, 6))
sns.countplot(x='flag', data=df, order=df['flag'].value_counts().index)
plt.title('Distribución de Estados de Conexión (Flags)')
plt.show()

# --- VISUALIZACIÓN 5: Mapa de Calor de Correlaciones (Variables Numéricas) ---
plt.figure(figsize=(15, 12))
# Filtramos solo columnas numéricas y eliminamos las que no tienen variación
numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1)
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0)
plt.title('Matriz de Correlación del Dataset Completo')
plt.show()

# --- VISUALIZACIÓN 6: Análisis de Bytes (Origen vs Destino) ---
plt.figure(figsize=(10, 6))
# Usamos escala logarítmica debido a la gran dispersión de los datos de red
plt.yscale('log')
plt.xscale('log')
sns.scatterplot(x='src_bytes', y='dst_bytes', hue='label', data=df, alpha=0.5)
plt.title('Relación Bytes Origen vs Destino (Escala Logarítmica)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
plt.show()

# --- VISUALIZACIÓN 7: Boxplot de Duración por Tipo de Protocolo ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='protocol_type', y='count', data=df)
plt.title('Distribución de Conexiones al mismo Host por Protocolo')
plt.show()