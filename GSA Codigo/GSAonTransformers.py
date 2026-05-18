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
from scipy.signal import butter, filtfilt
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
    X = train_tdah['X_tf']
    print(X.shape)
    return X



def extraer_features(X, threshold, local):
    feature = []

    for kk in range(X.shape[0]):
        graph, A, L = gpl.genero_grafo(
            X[kk, :, :], 
            threshold, 
            range(19), 
            [f'Ch{i}' for i in range(19)], 
            0
        )

        ec, spec_ratio, spec_gap, le, degree, ac = gpl.GSA(graph, A, L)
        densidad, g_clus, l_clus = gpl.calculo_grafo(graph, [(11, 7), (12, 16)])
        entropy = gpl.calculo_entropia(graph, degree)

        feat = {
            "AC": ac,
            "LE": le,
            "Entropy": entropy,
            "SR": spec_ratio,
            "SG": spec_gap,
            "dens": densidad,
            "g_clus": g_clus
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
            for j in range(i+1, n_channels):  # Solo triangulo superior (matriz simétrica)
                phase_diff = np.exp(1j * (phase_data[i, :] - phase_data[j, :]))  # Diferencia de fase en forma compleja
                plv_matrix[i, j] = np.abs(np.mean(phase_diff))  # PLV = media del módulo
                plv_matrix[j, i] = plv_matrix[i, j]  # Matriz simétrica
    
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

# Filtra un tensor de EEG (pacientes, canales, tiempo) en una banda específica.
def filtrar_banda_eeg(datos_3d, frec_min, frec_max, fs):
    nyq = 0.5 * fs
    if frec_max >= nyq:
        frec_max = nyq - 0.1
    low = frec_min / nyq
    high = frec_max / nyq
    b, a = butter(4, [low, high], btype='band')
    
    datos_filtrados = np.zeros_like(datos_3d)
    
    for i in range(datos_3d.shape[0]):
        for j in range(datos_3d.shape[1]):
            datos_filtrados[i, j, :] = filtfilt(b, a, datos_3d[i, j, :])
            
    return datos_filtrados

# %% [2] CARGA DE DATOS


########################## MAIN ###############################################

if __name__ == "__main__":
    
    print('#######################')
    print('EEG CRUDO + Transformer')
    print('#######################')

    eeg_channels = ["Fz","Cz","Pz","C3","T3","C4","T4","Fp1","Fp2","F3","F4","F7","F8","P3","P4","T5","T6","O1","O2"]
    channel_map = {}

    bandas = {
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta":  (13.0, 30.0),
        "Gamma":  (30.0, 100.0)
    }
    filtrar_por_banda = True

    for i, ch in enumerate(eeg_channels):
        channel_map[f"degree_{i}"] = f"d_{ch}"
        channel_map[f"ec_{i}"] = f"ec_{ch}"
 

    # Cargo datos
    fs = 128
    n_chs = 19
    
    inicio = time.time()  # Captura el tiempo de inicio
    X_ctrl_train = leodatos('../Transformer/train_class0_fold2.npz')
    X_ctrl_test = leodatos('../Transformer/test_class0_fold2.npz')
    X_tdah_train = leodatos('../Transformer/train_class1_fold2.npz')
    X_tdah_test = leodatos('../Transformer/test_class1_fold2.npz')
    # Inicializar las métricas con listas vacias

    if filtrar_por_banda:
        nombre_banda = "Gamma"
        f_min, f_max = bandas[nombre_banda]
        X_ctrl_train = filtrar_banda_eeg(X_ctrl_train, f_min, f_max, fs)
        X_ctrl_test = filtrar_banda_eeg(X_ctrl_test, f_min, f_max, fs)
        X_tdah_train = filtrar_banda_eeg(X_tdah_train, f_min, f_max, fs)
        X_tdah_test = filtrar_banda_eeg(X_tdah_test, f_min, f_max, fs)

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

    # %% [3] EXTRACCIÓN DE MÉTRICAS    

    threshold = 0.05   # Ojo, este threshold muy bajo porq las salidas del transformer estan descorrelacionadas
    top_frac = 0.20
    local = True
    threshold_local = False

    if threshold_local == False:
        threshold = umbral_global_train(plv_ctrl_train, plv_tdah_train, top_frac)
        print(f"Umbral global calculado: {threshold:.4f}")

    
    ctrl_feat_train  = pd.DataFrame(extraer_features(plv_ctrl_train , threshold, local))
    tdah_feat_train = pd.DataFrame(extraer_features(plv_tdah_train, threshold, local))
    ctrl_feat_test  = pd.DataFrame(extraer_features(plv_ctrl_test , threshold, local))
    tdah_feat_test = pd.DataFrame(extraer_features(plv_tdah_test, threshold, local))
    
    ctrl_feat_train["grupo"] = "CTRL"   
    ctrl_feat_train["split"] = "Train"   
    ctrl_feat_test["grupo"] = "CTRL"
    ctrl_feat_test["split"] = "test"
    tdah_feat_train["grupo"] = "TDAH"
    tdah_feat_train["split"] = "Train"
    tdah_feat_test["grupo"] = "TDAH"
    tdah_feat_test["split"] = "Test"

    if not local:    
         # Conjunto Train
         df_feat_train = pd.concat(
         [ctrl_feat_train, tdah_feat_train],
         ignore_index=True)
         X_train = df_feat_train [["dens","SR", "degree", "ec"]]
         y_train = (df_feat_train ["grupo"] == "TDAH").astype(int)
         
         # Conjunto Test
         df_feat_test = pd.concat(
         [ctrl_feat_test, tdah_feat_test],
         ignore_index=True)
         X_test = df_feat_test [["dens","SR", "degree", "ec"]]
         y_test = (df_feat_test ["grupo"] == "TDAH").astype(int)
     
    else:
         # Incorporando metricas regionales de GSA
         # Conjunto Train
         df_feat_train = pd.concat(
         [ctrl_feat_train, tdah_feat_train],
         ignore_index=True)
         X_train = df_feat_train.rename(columns=channel_map)
         X_train_model = X_train.drop(columns=["grupo", "split"])
         y_train = (df_feat_train ["grupo"] == "TDAH").astype(int)
     
         # Conjunto Test
         df_feat_test = pd.concat(
         [ctrl_feat_test, tdah_feat_test],
         ignore_index=True)
         X_test = df_feat_test.rename(columns=channel_map)
         X_test_model = X_test.drop(columns=["grupo", "split"])
         y_test = (df_feat_test ["grupo"] == "TDAH").astype(int)

    # %% [4] ENTRENAMIENTO XGBOOST

    plt.figure()
    model, metricas, importances = clasifico_Xgboost(X_train_model,y_train)
    plt.bar(X_train_model.columns, importances)
    plt.ylabel("Importance")
    plt.title("XGBoost feature importance")
    plt.show()
    
    # Evaluo el modelo con todas las features en datos de TEST
    y_todas_pred = model.predict(X_test_model)
    acc_test = accuracy_score(y_test, y_todas_pred)
    f1_test = f1_score(y_test, y_todas_pred)
    recall_test = recall_score(y_test, y_todas_pred)
    precision_test = precision_score(y_test, y_todas_pred)
    
    print('%%%%%%%%%%%%%% TEST (Raw EEG + Transformer) %%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'Accuracy: {acc_test:.4f}')
    print(f'F1: {f1_test:.4f}')
    print(f'Recall: {recall_test:.4f}')
    print(f'Precision: {precision_test:.4f}')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

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
    
    print(f'%%%%%%%%%%%%%% TEST (Raw EEG + Transformer) {n_largest_feat} features %%%%%%%%%%%%%%%%%%%%%%%%%%%%')
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
    plt.title("Raw EEG + Transformer")
    plt.show()
    
    # %% [6] GUARDAR RESULTADOS


    output_dir = '../Resultados/Transformers'
    parent_dir = '../Resultados/Transformers'

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
