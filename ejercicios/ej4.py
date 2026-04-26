# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =================================================================
# 1. GENERACIÓN DEL CONJUNTO DE DATOS
# =================================================================
# Generamos datos aleatorios para el ejemplo
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

print("La longitud del conjunto de datos es:", len(X))

# =================================================================
# 2. VISUALIZACIÓN DEL CONJUNTO DE DATOS
# =================================================================
plt.plot(X, y, "b.")
plt.xlabel("Equipos afectados (u/1000)")
plt.ylabel("Coste del incidente (u/10000)")
plt.show()

# =================================================================
# 3. MODIFICACIÓN Y PREPARACIÓN DEL CONJUNTO DE DATOS (PANDAS)
# =================================================================
data = {'n_equipos_afectados': X.flatten(), 'coste': y.flatten()}
df = pd.DataFrame(data)

# Escalado de los datos para simular valores reales
# El número de equipos se multiplica por 1000 y el coste por 10000
df['n_equipos_afectados'] = (df['n_equipos_afectados'] * 1000).astype('int')
df['coste'] = (df['coste'] * 10000).astype('int')

print("Primeras 10 filas del DataFrame:")
print(df.head(10))

# Representación gráfica de los datos escalados
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()

# =================================================================
# 4. CONSTRUCCIÓN DEL MODELO DE REGRESIÓN LINEAL
# =================================================================
# Inicializamos y entrenamos el modelo
lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)

# Parámetros obtenidos por el modelo (Theta 0 y Theta 1)
print("Intersección (Theta 0):", lin_reg.intercept_)
print("Coeficiente (Theta 1):", lin_reg.coef_[0])

# Predicción para la línea de regresión (mínimo y máximo)
X_min_max = np.array([[df["n_equipos_afectados"].min()], [df["n_equipos_afectados"].max()]])
y_train_pred = lin_reg.predict(X_min_max)

# Representación gráfica del modelo (línea de regresión)
plt.plot(X_min_max, y_train_pred, "g-", label="Línea de regresión")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.", label="Datos reales")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.legend()
plt.show()

# =================================================================
# 5. PREDICCIÓN DE NUEVOS EJEMPLOS
# =================================================================
# Predecimos el coste para un caso de 1300 equipos afectados
x_new = np.array([[1300]])
coste_predicho = lin_reg.predict(x_new)

print(f"El coste del incidente para {x_new[0][0]} equipos sería: {int(coste_predicho[0])} €")

# Visualización final con el nuevo punto predicho
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.plot(X_min_max, y_train_pred, "g-")
plt.plot(x_new, coste_predicho, "rx", markersize=15, label="Predicción")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.legend()
plt.show()