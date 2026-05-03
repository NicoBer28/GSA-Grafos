# %%  [1] LIBRERIAS Y FUNCIONES


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jan 11 10:55:35 2025

@author: mariapau
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import pickle
import Grafos_Paula_lib as gpl
import time
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score




def save_results(results, output_dir, arch):
    """Saves processed results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{arch}.pkl"), "wb") as f:
        pickle.dump(results, f)


def leodatos(path):
    train_tdah = np.load(path)
    print(train_tdah.files)
    X = train_tdah['X_in']
    print(X.shape)
    return X


def extraer_features(X, threshold, local, channels_str):
    def escalar_a_real(value):
        value = np.real_if_close(value)
        if np.iscomplexobj(value):
            value = np.abs(value)
        return float(value)

    def array_a_real(values):
        values = np.real_if_close(np.asarray(values))
        if np.iscomplexobj(values):
            values = np.abs(values)
        return values.astype(float)

    feature = []
    for kk in range(X.shape[0]):
    #for kk in range(15):
        #threshold = umbral_por_persona(X[kk, :, :], top_frac=0.20)
        #print(threshold)
        graph, A, L = gpl.genero_grafo(
            X[kk, :, :], 
            threshold, 
            range(19), 
            channels_str,
            0
        )

        ec, spec_ratio, spec_gap, le, degree, ac = gpl.GSA(graph, A, L)
        densidad, g_clus, l_clus = gpl.calculo_grafo(graph, [(11, 7), (12, 16)])
        entropy = gpl.calculo_entropia(graph, degree)

        ec = array_a_real(ec)
        degree = array_a_real(degree)

        feat = {
            "AC": escalar_a_real(abs(ac)),
            "LE": escalar_a_real(le),
            "Entropy": escalar_a_real(entropy),
            "SR": escalar_a_real(spec_ratio),
            "SG": escalar_a_real(abs(spec_gap)),
            "dens": escalar_a_real(densidad),
            "g_clus": escalar_a_real(g_clus)
        }

        
        if local:
            feat.update({f"degree_{i}": degree[i] for i in range(len(degree))})
            feat.update({f"ec_{i}": ec[i] for i in range(len(ec))})
        else:
            feat["degree"] = degree.mean()
            feat["ec"] = ec.mean()
        
        feature.append(feat)
    
    return feature



def clasifico_Xgboost(X,y):

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
     )
    cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
    )

    scoring = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": "f1"
    }
    
    cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
    )
    
    model.fit(X, y)    # entreno el modelo con todos los datos
    
    importances = model.feature_importances_
    
    return model, cv_results, importances



def phase_locking(datos, fs):
    
        """
        Calcula el Phase Locking Value (PLV) entre todos los pares de canales de una matriz EEG.
    
        Parámetros:
        - datos: array (n_canales, n_muestras) con las señales EEG filtradas en la banda de interés
        - sfreq: frecuencia de muestreo en Hz
        - fmin, fmax: rango de frecuencias de interés (ejemplo: 8-12 Hz para alfa)
    
        Retorna:
        - plv_matrix: matriz de PLV (n_canales, n_canales)
        """
        
        n_channels, n_samples = datos.shape
        plv_matrix = np.zeros((n_channels, n_channels))
    
        # Transformada de Hilbert para obtener la fase instantánea
        analytic_signal = signal.hilbert(datos, axis=1)
        phase_data = np.angle(analytic_signal)  # Extrae solo la fase
    
        # Calcular PLV entre todos los pares de canales
        for i in range(n_channels):
            for j in range(i+1, n_channels):  # Solo triangulo superior (matriz simétrica, las diagonales valen 0)
                phase_diff = np.exp(1j * (phase_data[i, :] - phase_data[j, :]))  # Diferencia de fase en forma compleja
                plv_matrix[i, j] = np.abs(np.mean(phase_diff))  # PLV = media del módulo
                plv_matrix[j, i] = plv_matrix[i, j]  # Matriz simétrica
        #np.fill_diagonal(plv_matrix, 1.0)
        return plv_matrix


def varios_modelos(X_flat,y):
    models = {
    "Logistic": LogisticRegression(max_iter=500),
    "SVM-RBF": SVC(kernel="rbf"),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        
        scores = cross_validate(
            pipe,
            X_flat,
            y,
            cv=cv,
            scoring=["accuracy", "recall", "precision"],
            n_jobs=-1
        )
        
        results[name] = {
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "recall_mean": scores["test_recall"].mean(),
            "precision_mean": scores["test_precision"].mean()
        }
    df_results = pd.DataFrame(results).T
    print(df_results.sort_values("accuracy_mean", ascending=False))

# Función para calcular el umbral global a partir de los datos de entrenamiento
def umbral_global_train(plv_ctrl_train, plv_tdah_train, top_frac):
    n_channels = plv_ctrl_train.shape[1]
    # Solo tomo los valores del triangulo superior de la matriz (sin incluir la diagonal)
    iu = np.triu_indices(n_channels, k=1)

    vals_ctrl = plv_ctrl_train[:, iu[0], iu[1]].ravel()
    vals_tdah = plv_tdah_train[:, iu[0], iu[1]].ravel()
    vals_train = np.concatenate([vals_ctrl, vals_tdah])

    return np.quantile(vals_train, 1 - top_frac)

# Función para calcular el umbral específico para cada persona a partir de su matriz PLV
def umbral_por_persona(plv_2d, top_frac):
    iu = np.triu_indices_from(plv_2d, k=1)
    vals = plv_2d[iu]
    return np.quantile(vals, 1 - top_frac)

# %% [2] CARGA DE DATOS

########################## MAIN ###############################################

if __name__ == "__main__":
    
    print('#######################')
    print('EEG CRUDO')
    print('#######################')

    eeg_channels = ["Fz","Cz","Pz","C3","T3","C4","T4","Fp1","Fp2","F3","F4","F7","F8","P3","P4","T5","T6","O1","O2"]
    channel_map = {}

    for i, ch in enumerate(eeg_channels):
        channel_map[f"degree_{i}"] = f"d_{ch}"
        channel_map[f"ec_{i}"] = f"ec_{ch}"
 
    
    # Cargo datos
    fs = 128
    n_chs = 19
    
    inicio = time.time()  # Captura el tiempo de inicio
    X_ctrl_train = leodatos('../EEG crudo/train_class0_fold2.npz')
    X_ctrl_test = leodatos('../EEG crudo/test_class0_fold2.npz')
    X_tdah_train = leodatos('../EEG crudo/train_class1_fold2.npz')
    X_tdah_test = leodatos('../EEG crudo/test_class1_fold2.npz')
    # Inicializar las métricas con listas vacias

    # Calculo PLV a cada registro
    plv_ctrl_train = np.zeros((X_ctrl_train.shape[0],n_chs,n_chs))
    plv_tdah_train = np.zeros((X_tdah_train.shape[0],n_chs,n_chs))
    plv_ctrl_test = np.zeros((X_ctrl_test.shape[0],n_chs,n_chs))
    plv_tdah_test = np.zeros((X_tdah_test.shape[0],n_chs,n_chs))
    
    for i in range(X_ctrl_train.shape[0]):
        plv_ctrl_train[i,:,:] = phase_locking(X_ctrl_train[i,:,:], fs)        
    for i in range(X_tdah_train.shape[0]):
        plv_tdah_train[i,:,:] = phase_locking(X_tdah_train[i,:,:], fs)
    for i in range(X_ctrl_test.shape[0]):
        plv_ctrl_test[i,:,:] = phase_locking(X_ctrl_test[i,:,:], fs)        
    for i in range(X_tdah_test.shape[0]):
        plv_tdah_test[i,:,:] = phase_locking(X_tdah_test[i,:,:], fs)
    # %%
    for i in range(X_ctrl_train.shape[0]):
        print(max(plv_ctrl_test[i::].ravel()))
        print(min(plv_ctrl_test[i::].ravel()))

    # %% [3] EXTRACCIÓN DE MÉTRICAS    

    threshold = umbral_global_train(plv_ctrl_train, plv_tdah_train, top_frac=0.20)
    #threshold = 0.4   
    local = True
    #print(threshold)


    ctrl_feat_train  = pd.DataFrame(extraer_features(plv_ctrl_train , threshold, local, channels_str=eeg_channels))
    tdah_feat_train = pd.DataFrame(extraer_features(plv_tdah_train, threshold, local, channels_str=eeg_channels))
    ctrl_feat_test  = pd.DataFrame(extraer_features(plv_ctrl_test , threshold, local, channels_str=eeg_channels))
    tdah_feat_test = pd.DataFrame(extraer_features(plv_tdah_test, threshold, local, channels_str=eeg_channels))


    ctrl_feat_train["grupo"] = "CTRL"   
    ctrl_feat_train["split"] = "Train"   
    ctrl_feat_test["grupo"] = "CTRL"
    ctrl_feat_test["split"] = "test"
    tdah_feat_train["grupo"] = "TDAH"
    tdah_feat_train["split"] = "Train"
    tdah_feat_test["grupo"] = "TDAH"
    tdah_feat_test["split"] = "Test"

    # "AC"
    # "LE"
    # "Entropy"
    # "SR"
    # "SG"
    # "dens"
    # "g_clus"
    # "degree"
    # "ec"

    if not local:    
        # Conjunto Train
        df_feat_train = pd.concat(
        [ctrl_feat_train, tdah_feat_train],
        ignore_index=True)
        X_train = df_feat_train [["dens","SR", "degree", "ec"]]
        #X_train = df_feat_train [["AC","Entropy", "g_clus", "degree", "ec"]]
        y_train = (df_feat_train ["grupo"] == "TDAH").astype(int)
        
        # Conjunto Test
        df_feat_test = pd.concat(
        [ctrl_feat_test, tdah_feat_test],
        ignore_index=True)
        X_test = df_feat_test [["dens","SR", "degree", "ec"]]
        #X_test = df_feat_test [["AC","Entropy", "g_clus", "degree", "ec"]]
        y_test = (df_feat_test ["grupo"] == "TDAH").astype(int)
    
    else:
        # Incorporando metricas regionales de GSA
        # Conjunto Train
        df_feat_train = pd.concat(
        [ctrl_feat_train, tdah_feat_train],
        ignore_index=True)
        X_train = df_feat_train.rename(columns=channel_map)
        y_train = (df_feat_train ["grupo"] == "TDAH").astype(int)
    
        # Conjunto Test
        df_feat_test = pd.concat(
        [ctrl_feat_test, tdah_feat_test],
        ignore_index=True)
        X_test = df_feat_test.rename(columns=channel_map)
        y_test = (df_feat_test ["grupo"] == "TDAH").astype(int)

    X_train_model = X_train.drop(columns=["grupo", "split"], errors="ignore")
    X_test_model = X_test.drop(columns=["grupo", "split"], errors="ignore")

    # %% [4] ENTRENAMIENTO XGBOOST

    plt.figure()
    model, metricas, importances = clasifico_Xgboost(X_train_model,y_train)
    plt.bar(X_train_model.columns, importances)
    plt.ylabel("Importance")
    plt.title("Raw EEG")
    plt.show()
    
    # Evaluo el modelo con todas las features en datos de TEST
    y_todas_pred = model.predict(X_test_model)
    acc_test = accuracy_score(y_test, y_todas_pred)
    f1_test = f1_score(y_test, y_todas_pred)
    recall_test = recall_score(y_test, y_todas_pred)
    precision_test = precision_score(y_test, y_todas_pred)
    
    print('%%%%%%%%%%%%%% TEST (Raw EEG) %%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'Accuracy: {acc_test:.4f}')
    print(f'F1: {f1_test:.4f}')
    print(f'Recall: {recall_test:.4f}')
    print(f'Precision: {precision_test:.4f}')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

# %%
    for i, imp in enumerate(importances):
        print(f"Feature {i}: {imp:.4f}")
    # %% [5] ENTRENAMIENTO CON TOP N FEATURES

    plt.figure()
    n_largest_feat = 6
    top_n_cols = pd.Series(importances, index=X_train_model.columns).nlargest(n_largest_feat).index
    X_top_n_train = X_train_model[top_n_cols]
    X_top_n_test  = X_test_model[top_n_cols]
    # obtengo el modelo (sin entrenar) y métricas CV
    model, metricas, importances = clasifico_Xgboost(X_top_n_train,y_train)
    
    # Evaluación final en test (usar X_test,y_test reales)
    y_pred = model.predict(X_top_n_test)
    y_prob = model.predict_proba(X_top_n_test)[:,1] 
    
   
    acc_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    recall_test = recall_score(y_test, y_pred)     
    precision_test = precision_score(y_test, y_pred)
    auc_test = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    print(f'%%%%%%%%%%%%%% TEST (Raw EEG) {n_largest_feat} features %%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'Accuracy: {acc_test:.4f}')
    print(f'F1: {f1_test:.4f}')
    print(f'Recall: {recall_test:.4f}')
    print(f'Precision: {precision_test:.4f}')
    if auc_test is not None:
        print(f'AUC ROC: {auc_test:.4f}')
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(top_n_cols)
    plt.bar(X_top_n_test.columns, importances)
    plt.ylabel("Importance")
    plt.title("Raw EEG")
    plt.show()

    # %% [6] GUARDAR RESULTADOS

    output_dir = '../Resultados/EEG'

    fin = time.time()  # Captura el tiempo de inicio
    print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")

    # Genero las matrices de conectividad (PLV) promedio
    # para todos los grupos
    
    plv_ctrl_train_prom = np.mean(plv_ctrl_train, axis = 0)
    plv_ctrl_test_prom = np.mean(plv_ctrl_test, axis = 0)
    plv_tdah_train_prom = np.mean(plv_tdah_train, axis = 0)
    plv_tdah_test_prom = np.mean(plv_tdah_test, axis = 0)
    
    save_results(plv_ctrl_train_prom, output_dir, 'plv_ctrl_train_prom')
    save_results(plv_ctrl_test_prom, output_dir,'plv_ctrl_test_prom')
    save_results(plv_tdah_train_prom, output_dir, 'plv_tdah_train_prom')
    save_results(plv_tdah_test_prom, output_dir, 'plv_tdah_test_prom')
    
# %%
