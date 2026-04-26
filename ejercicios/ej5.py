import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =================================================================
# 1. CARGA Y PREPARACIÓN DEL CSV
# =================================================================
try:
    # Cargamos el archivo. Si usa otro separador, añade sep=';'
    df = pd.read_csv('processed_data.csv') 
    
    # Limpieza básica: eliminamos filas donde falte el mensaje o la etiqueta
    df = df.dropna(subset=['message', 'label'])
    
    # Combinar Asunto y Mensaje para mejor precisión
    X_raw = df['subject'].fillna('') + " " + df['message']
    
    y = df['label']
    
    print(f"Dataset cargado: {df.shape[0]} registros.")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# =================================================================
# 2. VECTORIZACIÓN (NLP BÁSICO)
# =================================================================
# Convertimos el texto a una matriz de conteo de palabras
vectorizer = CountVectorizer(stop_words='english') # Filtra palabras comunes
X = vectorizer.fit_transform(X_raw)

# División 80/20 para validar el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =================================================================
# 3. ENTRENAMIENTO (REGRESIÓN LOGÍSTICA)
# =================================================================
# La regresión logística clasifica basándose en probabilidades
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, y_train)

# =================================================================
# 4. EVALUACIÓN
# =================================================================
y_pred = model.predict(X_test)

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))

# =================================================================
# 5. TEST DE PREDICCIÓN
# =================================================================
def test_mensaje(texto):
    vec = vectorizer.transform([texto])
    prob = model.predict_proba(vec)[0][1] # Probabilidad de ser clase 1 (SPAM)
    print(f"Texto: {texto[:50]}...")
    print(f"Probabilidad de SPAM: {prob:.2%}")

print("\nPrueba de predicción:")
test_mensaje("Urgent: your account has been compromised, click here")