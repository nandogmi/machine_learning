#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# CREACIÓN DE TRANSFORMADORES Y PIPELINES PERSONALIZADOS
# =============================================================================
# En este script se muestra la creación de transformadores y Pipelines personalizados
#
# CONJUNTO DE DATOS
# -----------------
# Descripción:
# NSL-KDD is a data set suggested to solve some of the inherent problems of the KDD'99 
# data set. It can be applied as an effective benchmark data set to help researchers 
# compare different intrusion detection methods.
#
# Ficheros de datos principales:
# * KDDTrain+.ARFF: The full NSL-KDD train set with binary labels in ARFF format
#
# Referencias adicionales sobre el conjunto de datos:
# M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set”
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def load_kdd_dataset(data_path):
    """Lectura del conjunto de datos NSL-KDD."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=attributes)

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

# =============================================================================
# LECTURA Y DIVISIÓN DEL CONJUNTO DE DATOS
# =============================================================================
# Nota: Asegúrate de tener el archivo en la ruta 'datasets/NSL-KDD/KDDTrain+.arff'
df = load_kdd_dataset("KDDTrain+.arff")

# División del conjunto de datos
train_set, val_set, test_set = train_val_test_split(df, stratify='protocol_type')

print("Longitud del Training Set:", len(train_set))
print("Longitud del Validation Set:", len(val_set))
print("Longitud del Test Set:", len(test_set))

# =============================================================================
# API SKLEARN (Reseña teórica)
# =============================================================================
# * Estimators: Cualquier objeto que puede estimar algún parámetro.
#   - Se forma mediante el método fit() (toma un dataset como argumento).
#   - Otros parámetros de este método son hiperparámetros.
# * Transformers: Son estimadores capaces de transformar el conjunto de datos.
#   - La transformación se realiza mediante el método transform().
# * Predictors: Son estimadores capaces de realizar predicciones.
#   - La predicción se realiza mediante predict().
#   - Tienen un método score() para evaluar la predicción.

# =============================================================================
# 1. CONSTRUYENDO TRANSFORMADORES PERSONALIZADOS
# =============================================================================
# La creación de transformadores propios permite mantener el código limpio y 
# estructurado al preparar datos para algoritmos de ML, facilitando su reutilización.

# Separamos las etiquetas del resto de los datos
X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

# Añadimos valores nulos para ilustrar el funcionamiento de los transformadores
X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"]<800), "src_bytes"] = np.nan
X_train.loc[(X_train["dst_bytes"]>500) & (X_train["dst_bytes"]<2000), "dst_bytes"] = np.nan

# --- Transformadores para atributos numéricos ---

from sklearn.base import BaseEstimator, TransformerMixin

# 1. Transformador creado para eliminar las filas con valores nulos
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.dropna()

delete_nan = DeleteNanRows()
X_train_prep = delete_nan.fit_transform(X_train)


# 2. Transformador diseñado para escalar únicamente unas columnas seleccionadas
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        return X_copy

custom_scaler = CustomScaler(["src_bytes", "dst_bytes"])
X_train_prep = custom_scaler.fit_transform(X_train_prep)


# 3. Transformador para codificar únicamente las columnas categóricas y devolver un DataFrame
# Nota: Requiere importar OneHotEncoder (se importa más abajo en el original)
from sklearn.preprocessing import OneHotEncoder

class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._oh = OneHotEncoder(sparse_output=False) # sparse_output en versiones recientes de sklearn
        self._columns = None
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])
        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh = pd.DataFrame(X_cat_oh, 
                                columns=self._columns, 
                                index=X_copy.index)
        X_copy.drop(list(X_cat), axis=1, inplace=True)
        return X_copy.join(X_cat_oh)

# =============================================================================
# 6. CONSTRUYENDO PIPELINES PERSONALIZADOS
# =============================================================================
# Las pipelines agrupan en un flujo de ejecución todas las operaciones de 
# transformación. 
# * Todos menos el último componente deben ser transformadores.
# * Se llama secuencialmente a fit_transform() pasando el output al siguiente.

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Construcción de un pipeline para los atributos numéricos
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('rbst_scaler', RobustScaler()),
    ])

# La clase imputer no admite valores categóricos, procesamos solo los numéricos
X_train_num = X_train.select_dtypes(exclude=['object'])

X_train_prep_num = num_pipeline.fit_transform(X_train_num)
X_train_prep_num = pd.DataFrame(X_train_prep_num, columns=X_train_num.columns, index=X_train_num.index)

# ColumnTransformer ejecuta todos los pipelines en paralelo y concatena el resultado
from sklearn.compose import ColumnTransformer

num_attribs = list(X_train.select_dtypes(exclude=['object']))
cat_attribs = list(X_train.select_dtypes(include=['object']))

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

# Ejecución del pipeline completo sobre los datos originales
X_train_prep_full = full_pipeline.fit_transform(X_train)

# Reconstrucción del DataFrame final para visualización
X_train_prep_full = pd.DataFrame(X_train_prep_full, columns=list(pd.get_dummies(X_train)), index=X_train.index)

# Imprimir una muestra para verificar
# print(X_train_prep_full.head(10))