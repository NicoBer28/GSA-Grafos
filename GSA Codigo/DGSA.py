# %%  [1] LIBRERIAS Y FUNCIONES


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import importlib


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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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


def leo_y_divido_datos(ruta_npz, test_size=0.2, random_state=42):
    """
    Carga un archivo .npz unificado y devuelve las 4 matrices
    """
    """
    TE_matrices: tensor 3D con las 1659 ventanas (trials) de 19x19.
    subject_ids: Un vector (1659 elementos) que  dice a qué paciente pertenece cada una de esas ventanas.
    class_names: Un vector (1659 elementos) que indica a qué clase (Control o TDAH)pertenece cada una de esas ventanas.
    """
    # cargar el archivo completo
    print(f"Cargando archivo: {ruta_npz}")
    data = np.load(ruta_npz, allow_pickle=True)
    
    TE_matrices = data["TE_matrices"]
    subject_ids = data["subject_ids"].astype(str)
    class_names = data["class_names"].astype(str)
    
    # identificar los sujetos únicos que pertenecen a cada clase
    sujetos_ctrl = np.unique(subject_ids[class_names == 'Control'])
    sujetos_tdah = np.unique(subject_ids[class_names == 'TDAH'])
    
    # dividir los sujetos en grupos de train y test
    train_subj_ctrl, test_subj_ctrl = train_test_split(sujetos_ctrl, test_size=test_size, random_state=random_state)
    train_subj_tdah, test_subj_tdah = train_test_split(sujetos_tdah, test_size=test_size, random_state=random_state)


    # crear máscaras booleanas buscando a qué grupo pertenece cada trial
    # isin devuelve un vector booleano del mismo tamaño que subject_ids, con True donde el subject_id está en el grupo correspondiente
    mask_ctrl_train = np.isin(subject_ids, train_subj_ctrl)
    mask_ctrl_test = np.isin(subject_ids, test_subj_ctrl)
    
    mask_tdah_train = np.isin(subject_ids, train_subj_tdah)
    mask_tdah_test = np.isin(subject_ids, test_subj_tdah)
    
    # extraer las matrices finales usando las máscaras
    # te quedas solo con los trials que corresponden a cada grupo
    X_ctrl_train = TE_matrices[mask_ctrl_train]
    X_ctrl_test = TE_matrices[mask_ctrl_test]
    
    X_tdah_train = TE_matrices[mask_tdah_train]
    X_tdah_test = TE_matrices[mask_tdah_test]
    
    
    return X_ctrl_train, X_ctrl_test, X_tdah_train, X_tdah_test


def extraer_features_dirigidas(
    X,
    threshold,
    local,
    q,
    threshold_local=False,
    top_frac=0.5,
    channels_str=None,
):
    feature = []
    
    for kk in range(X.shape[0]):
        if threshold_local:
            threshold = umbral_por_persona(X[kk, :, :], top_frac)
            
        graph, A, L = gpl.genero_grafo_dirigido(
            X[kk, :, :], 
            threshold, 
            range(X.shape[1]),
            channels_str,
            q,
            ploteo=0,
        )
        
        ac, spec_gap, ec_magnitud, ec_fase = gpl.DGSA(L)
        densidad, g_clus, l_clus, reciprocidad, in_degree, out_degree, flujo, pr = gpl.calculo_grafo_dirigido(graph, A)
        

        feat = {
            "AC_mag": ac,
            "SG_mag": spec_gap,
            "dens_dir": densidad,
            "g_clus_dir": g_clus,
            "reciprocidad": reciprocidad
        }
    
        if local:
            for i in range(len(channels_str)):
                feat[f"in_degree_{i}"] = in_degree[i]
                feat[f"out_degree_{i}"] = out_degree[i]
                feat[f"flow_{i}"] = flujo[i]
                feat[f"pagerank_{i}"] = pr[i]
                feat[f"ec_mag_{i}"] = ec_magnitud[i]
                feat[f"ec_fase_{i}"] = ec_fase[i]
                feat[f"l_clus_{i}"] = l_clus[i]
        else:
            feat["in_degree"] = in_degree.mean()
            feat["out_degree"] = out_degree.mean()
            feat["flow"] = flujo.mean()
            feat["pagerank"] = pr.mean()
            feat["ec_mag"] = ec_magnitud.mean()
            feat["ec_fase"] = ec_fase.mean()
            feat["l_clus"] = l_clus.mean()

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
def umbral_global_train_dirigido(te_ctrl_train, te_tdah_train, top_frac):
    # Aplanamos las matrices completas
    vals_ctrl = te_ctrl_train.flatten()
    vals_tdah = te_tdah_train.flatten()
    vals_train = np.concatenate([vals_ctrl, vals_tdah])
    
    # Quitamos los ceros (la diagonal y las autoconexiones) para no sesgar el cuantil
    vals_train = vals_train[vals_train > 0]
    
    return np.quantile(vals_train, 1 - top_frac)

# Función para calcular el umbral específico para cada persona a partir de su matriz PLV
def umbral_por_persona(plv_2d, top_frac):
    iu = np.triu_indices_from(plv_2d, k=1)
    vals = plv_2d[iu]
    return np.quantile(vals, 1 - top_frac)

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
    print('EEG CRUDO')
    print('#######################')

    eeg_channels = ["Fz","Cz","Pz","C3","T3","C4","T4","Fp1","Fp2","F3","F4","F7","F8","P3","P4","T5","T6","O1","O2"]
    channel_map = {}


    for i, ch in enumerate(eeg_channels):
        channel_map[f"in_degree_{i}"] = f"in_d_{ch}"
        channel_map[f"out_degree_{i}"] = f"out_d_{ch}"
        channel_map[f"flow_{i}"] = f"flow_{ch}"
        channel_map[f"pagerank_{i}"] = f"pr_{ch}"
        channel_map[f"ec_mag_{i}"] = f"ec_mag_{ch}"
        channel_map[f"ec_fase_{i}"] = f"ec_fase_{ch}"
        channel_map[f"l_clus_{i}"] = f"l_clus_{ch}"
 
    
    # Cargo datos
    fs = 128
    n_chs = 19
    
    inicio = time.time()  # Captura el tiempo de inicio
    ruta_archivo = "../Entropia/TE_arrays_test_best_fold_2_diag_zero.npz"

    X_ctrl_train, X_ctrl_test, X_tdah_train, X_tdah_test = leo_y_divido_datos(
        ruta_npz = ruta_archivo, 
        test_size = 0.2,   # cambiar aca para elegir el tamaño del test
        random_state = 42  # semilla fija para que no divida de forma distinta cada vez que se ejecute el codigo
    )

    # print('--- Rango de valores en los datos ---')
    # print(f"Valor mínimo en CTRL Train: {np.min(X_ctrl_train):.4f}")
    # print(f"Valor máximo en CTRL Train: {np.max(X_ctrl_train):.4f}")
    # print(f"Valor mínimo en TDAH Train: {np.min(X_tdah_train):.4f}")
    # print(f"Valor máximo en TDAH Train: {np.max(X_tdah_train):.4f}")
    # print('-------------------------------------')
    
    # np.set_printoptions(precision=3, suppress=True, linewidth=120)

    # print("\nMatriz TE (19x19) del PRIMER trial de Control:")
    # print(X_ctrl_train[0])

    # %% [3] EXTRACCIÓN DE MÉTRICAS    
    threshold = 0.9   
    top_frac = 0.20
    local = True
    threshold_local = False
    grafo_complementario = False
    q=0.1
    # La variable q es un parámetro de "carga" que vos elegís
    # (suele ser un número chico como 0.1 o 0.25)
    # para darle más o menos importancia a la direccionalidad de las flechas.

    #print(threshold)
    if threshold_local == False:
        threshold = umbral_global_train_dirigido(X_ctrl_train, X_tdah_train, top_frac)
        print(f"Umbral global calculado: {threshold:.4f}")

    ctrl_feat_train  = pd.DataFrame(extraer_features_dirigidas(X_ctrl_train , threshold, local, q, threshold_local, top_frac, channels_str=eeg_channels))
    tdah_feat_train = pd.DataFrame(extraer_features_dirigidas(X_tdah_train, threshold, local, q, threshold_local, top_frac, channels_str=eeg_channels))
    ctrl_feat_test  = pd.DataFrame(extraer_features_dirigidas(X_ctrl_test , threshold, local, q, threshold_local, top_frac, channels_str=eeg_channels))
    tdah_feat_test = pd.DataFrame(extraer_features_dirigidas(X_tdah_test, threshold, local, q, threshold_local, top_frac, channels_str=eeg_channels))


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

    output_dir = '../Resultados/Entropia'

    fin = time.time()  # Captura el tiempo de inicio
    print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")

    # Genero las matrices de conectividad (TE) promedio para todos los grupos
    
    te_ctrl_train_prom = np.mean(X_ctrl_train, axis = 0)
    te_ctrl_test_prom = np.mean(X_ctrl_test, axis = 0)
    te_tdah_train_prom = np.mean(X_tdah_train, axis = 0)
    te_tdah_test_prom = np.mean(X_tdah_test, axis = 0)
    
    save_results(te_ctrl_train_prom, output_dir, 'te_ctrl_train_prom')
    save_results(te_ctrl_test_prom, output_dir,'te_ctrl_test_prom')
    save_results(te_tdah_train_prom, output_dir, 'te_tdah_train_prom')
    save_results(te_tdah_test_prom, output_dir, 'te_tdah_test_prom')
  
# %%
