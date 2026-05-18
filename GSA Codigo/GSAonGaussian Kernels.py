# %%  [1] LIBRERIAS Y FUNCIONES

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:55:35 2025

@author: mariapau
"""

import numpy as np
import pandas as pd
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



def save_results(results, output_dir):
    """Saves processed results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    for key, value in results.items():
        with open(os.path.join(output_dir, f"{key}.pkl"), "wb") as f:
            pickle.dump(pd.DataFrame(value), f)

def leodatos(path):
    data = np.load(path)
    print(data.files)
    X = data['gaussian_layer_1']
    X = np.squeeze(X, axis = -1)
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

# %% [2] CARGA DE DATOS

########################## MAIN ###############################################

if __name__ == "__main__":
    
    print('#######################')
    print('EEG CRUDO + Transformer + GK')
    print('#######################')

    eeg_channels = ["Fz","Cz","Pz","C3","T3","C4","T4","Fp1","Fp2","F3","F4","F7","F8","P3","P4","T5","T6","O1","O2"]
    channel_map = {}

    for i, ch in enumerate(eeg_channels):
        channel_map[f"degree_{i}"] = f"d_{ch}"
        channel_map[f"ec_{i}"] = f"ec_{ch}"
 


    n_chs = 19
    inicio = time.time()  # Captura el tiempo de inicio
    X_ctrl_train = leodatos('../Gausian Kernel/train_class0_fold2.npz')
    X_ctrl_test = leodatos('../Gausian Kernel/test_class0_fold2.npz')
    X_tdah_train = leodatos('../Gausian Kernel/train_class1_fold2.npz')
    X_tdah_test = leodatos('../Gausian Kernel/test_class1_fold2.npz')
    # Inicializar las métricas con listas vacias

    # for j in range(X_ctrl_train.shape[0]):
    #     X_ctrl_train[j,:,:] = X_ctrl_train[j,:,:]  - np.eye(n_chs,n_chs)
    # for j in range(X_ctrl_test.shape[0]):
    #     X_ctrl_test[j,:,:] = X_ctrl_test[j,:,:]  - np.eye(n_chs,n_chs)
    # for j in range(X_tdah_train.shape[0]):
    #     X_tdah_train[j,:,:] = X_tdah_train[j,:,:]  - np.eye(n_chs,n_chs)
    # for j in range(X_tdah_test.shape[0]):
    #     X_tdah_test[j,:,:] = X_tdah_test[j,:,:]  - np.eye(n_chs,n_chs)

    for dataset in (X_ctrl_train, X_ctrl_test, X_tdah_train, X_tdah_test):
        for j in range(dataset.shape[0]):
            np.fill_diagonal(dataset[j], 0)

    # %% [3] EXTRACCIÓN DE MÉTRICAS    

    threshold = 0.92   
    top_frac = 0.50
    local = True
    threshold_local = False    

    if threshold_local == False:
        threshold = umbral_global_train(X_ctrl_train, X_tdah_train, top_frac)
        print(f"Umbral global calculado: {threshold:.4f}")

    ctrl_train_feat = pd.DataFrame(extraer_features(X_ctrl_train, threshold, local))
    ctrl_test_feat = pd.DataFrame(extraer_features(X_ctrl_test, threshold, local))
    tdah_train_feat = pd.DataFrame(extraer_features(X_tdah_train, threshold, local))
    tdah_test_feat  = pd.DataFrame(extraer_features(X_tdah_test, threshold, local))
    
    ctrl_train_feat["grupo"] = "CTRL"
    ctrl_train_feat["split"] = "train"
    
    ctrl_test_feat["grupo"] = "CTRL"
    ctrl_test_feat["split"] = "test"
    
    tdah_train_feat["grupo"] = "TDAH"
    tdah_train_feat["split"] = "train"
    
    tdah_test_feat["grupo"] = "TDAH"
    tdah_test_feat["split"] = "test"

    if not local:    
        # Conjunto Train
        df_feat_train = pd.concat(
        [ctrl_train_feat, tdah_train_feat],
        ignore_index=True)
        X_train = df_feat_train [["dens","SR", "degree", "ec"]]
        y_train = (df_feat_train ["grupo"] == "TDAH").astype(int)
        
        # Conjunto Test
        df_feat_test = pd.concat(
        [ctrl_test_feat, tdah_test_feat],
        ignore_index=True)
        X_test = df_feat_test [["dens","SR", "degree", "ec"]]
        y_test = (df_feat_test ["grupo"] == "TDAH").astype(int)
    
    else:
        # Incorporando metricas regionales de GSA
        # Conjunto Train
        df_feat_train = pd.concat(
        [ctrl_train_feat, tdah_train_feat],
        ignore_index=True)
        X_train = df_feat_train.rename(columns=channel_map)
        X_train_model = X_train.drop(columns=["grupo", "split"])
        y_train = (df_feat_train ["grupo"] == "TDAH").astype(int)
    
        # Conjunto Test
        df_feat_test = pd.concat(
        [ctrl_test_feat, tdah_test_feat],
        ignore_index=True)
        X_test = df_feat_test.rename(columns=channel_map)
        X_test_model = X_test.drop(columns=["grupo", "split"])
        y_test = (df_feat_test ["grupo"] == "TDAH").astype(int)

    # %% [4] ENTRENAMIENTO XGBOOST


    plt.figure()
    model, metricas, importances = clasifico_Xgboost(X_train_model,y_train)
    plt.bar(X_train_model.columns, importances)
    plt.ylabel("Importance")
    plt.title("Raw EEG + Transformer + GK")
    plt.show()
    
    # Evaluo el modelo con todas las features en datos de TEST
    y_todas_pred = model.predict(X_test_model)
    acc_test = accuracy_score(y_test, y_todas_pred)
    f1_test = f1_score(y_test, y_todas_pred)
    recall_test = recall_score(y_test, y_todas_pred)
    precision_test = precision_score(y_test, y_todas_pred)
    
    print('%%%%%%%%%%%%%% TEST (todas las features) %%%%%%%%%%%%%%%%%%%%%%%%%%%%')
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
    
    print(f'%%%%%%%%%%%%%% TEST (final) {n_largest_feat} features %%%%%%%%%%%%%%%%%%%%%%%%%%%%')
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
    plt.title("Raw EEG + Transformer + GK")
    plt.show()

    # %% [6] GUARDAR RESULTADOS

    output_dir = '../Resultados/GaussianKernels'
    parent_dir = '../Resultados/GaussianKernels'

    fin = time.time()  # Captura el tiempo de inicio
    print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")

        
   