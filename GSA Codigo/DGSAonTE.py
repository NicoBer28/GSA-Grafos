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
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


def cargar_datos_split(ruta_base, fold, split):
    """
    Entra a fold_X/split/ y carga todas las matrices separando por TDAH y Control.
    Devuelve: X (matrices apiladas), y (etiquetas), ids (nombre del paciente)
    """
    ruta_split = os.path.join(ruta_base, f"fold_{fold}", split)
    
    matrices_list = []
    labels_list = []
    ids_list = []
    
    # Recorremos ambas carpetas (0 para Control, 1 para TDAH)
    for clase, label in [("Control", 0), ("TDAH", 1)]:
        ruta_clase = os.path.join(ruta_split, clase)
        
        # Si la carpeta no existe, la saltamos (por seguridad)
        if not os.path.exists(ruta_clase): 
            continue
            
        for archivo in os.listdir(ruta_clase):
            if archivo.endswith(".npz"):
                paciente_id = archivo.replace(".npz", "")
                
                # Cargamos el archivo del paciente
                data = np.load(os.path.join(ruta_clase, archivo))
                
                # Extraemos el tensor (tomamos la primera variable que haya adentro del npz)
                key = data.files[0] 
                matrices = data[key] # Forma: (n_ventanas, 19, 19)
                
                matrices_list.append(matrices)
                labels_list.extend([label] * matrices.shape[0])
                ids_list.extend([paciente_id] * matrices.shape[0])
                
    if len(matrices_list) > 0:
        X = np.concatenate(matrices_list, axis=0)
        y = np.array(labels_list)
        ids = np.array(ids_list)
        return X, y, ids
    else:
        return np.array([]), np.array([]), np.array([])

#-----LOSO-----
def validacion_LOSO_por_paciente(X_features, y_labels, groups_ids):
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    
    # vectores globales para VENTANAS 
    y_reales_ventanas = []
    y_pred_ventanas = []
    
    # vectores globales para PACIENTES
    y_reales_paciente = []
    y_pred_paciente = []
    
    X_arr = X_features.values if isinstance(X_features, pd.DataFrame) else X_features
    y_arr = y_labels.values if isinstance(y_labels, pd.Series) else y_labels
    groups_arr = groups_ids.values if isinstance(groups_ids, pd.Series) else groups_ids
    
    for train_idx, test_idx in logo.split(X_arr, y_arr, groups_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        
        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # 1. Evaluación por Ventana
        predicciones_del_sujeto = model.predict(X_test)
        y_reales_ventanas.extend(y_test)
        y_pred_ventanas.extend(predicciones_del_sujeto)
        
        # 2. Evaluación por Paciente (Votación Mayoritaria)
        # Si más del 50% de las ventanas dice TDAH (1), diagnosticamos TDAH
        if np.mean(predicciones_del_sujeto) > 0.5:
            voto_final = 1
        else:
            voto_final = 0
            
        # Guardamos el voto del paciente y su etiqueta real (miramos la primera ventana)
        y_reales_paciente.append(y_test[0])
        y_pred_paciente.append(voto_final)
        
    # --- RESULTADOS FINALES ---
    print('\n' + '='*50)
    print(' RENDIMIENTO CLÍNICO (POR PACIENTE)')
    print('='*50)
    print(f'Accuracy : {accuracy_score(y_reales_paciente, y_pred_paciente):.4f}')
    print(f'F1-Score : {f1_score(y_reales_paciente, y_pred_paciente):.4f}')
    print(f'Recall   : {recall_score(y_reales_paciente, y_pred_paciente):.4f}')
    print(f'Precision: {precision_score(y_reales_paciente, y_pred_paciente):.4f}')
    
    print('\n' + '-'*50)
    print(' RENDIMIENTO ALGORÍTMICO (POR VENTANA)')
    print('-'*50)
    print(f'Accuracy : {accuracy_score(y_reales_ventanas, y_pred_ventanas):.4f}')
    print(f'F1-Score : {f1_score(y_reales_ventanas, y_pred_ventanas):.4f}')
    print(f'Recall   : {recall_score(y_reales_ventanas, y_pred_ventanas):.4f}')
    print(f'Precision: {precision_score(y_reales_ventanas, y_pred_ventanas):.4f}')
    
    # Entrenamos el modelo final para sacar las importancias
    model_final = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_final.fit(X_arr, y_arr)
    importances = model_final.feature_importances_
    
    return model_final, importances

def validacion_LOSO(X_features, y_labels, groups_ids):
    logo = LeaveOneGroupOut()
    
    # En lugar de guardar métricas, guardamos las etiquetas reales y predichas de TODOS
    y_reales_global = []
    y_pred_global = []
    
    X_arr = X_features.values if isinstance(X_features, pd.DataFrame) else X_features
    y_arr = y_labels.values if isinstance(y_labels, pd.Series) else y_labels
    groups_arr = groups_ids.values if isinstance(groups_ids, pd.Series) else groups_ids
    
    for train_idx, test_idx in logo.split(X_arr, y_arr, groups_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        
        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # AGREGAMOS las predicciones de este paciente a la lista global
        y_reales_global.extend(y_test)
        y_pred_global.extend(y_pred)
        
    # --- CÁLCULO DE MÉTRICAS GLOBALES ---
    # Una vez que pasaron todos los pacientes, evaluamos el desempeño general
    print('%%%%%%%%%%%%%% RESULTADOS LOSO (GLOBALES) %%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'Accuracy Global: {accuracy_score(y_reales_global, y_pred_global):.4f}')
    print(f'F1 Global: {f1_score(y_reales_global, y_pred_global):.4f}')
    print(f'Recall Global: {recall_score(y_reales_global, y_pred_global):.4f}')
    print(f'Precision Global: {precision_score(y_reales_global, y_pred_global):.4f}')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    # Entrenamos el modelo final para sacar las importancias
    model_final = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model_final.fit(X_arr, y_arr)
    importances = model_final.feature_importances_
    
    return model_final, importances


def save_results(results, output_dir, arch):
    """Saves processed results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{arch}.pkl"), "wb") as f:
        pickle.dump(results, f)



def extraer_features(
    X,
    threshold,
    local,
    threshold_local=False,
    grafo_complementario=False,
    top_frac=0.5,
    channels_str=None,
):
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
        if threshold_local:
            threshold = umbral_por_persona(X[kk, :, :], top_frac)
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


        feat = {
            "AC": abs(ac),
            "LE": le,
            "Entropy": entropy,
            "SR": spec_ratio,
            "SG": abs(spec_gap),
            "dens": densidad,
            "g_clus": g_clus
        }
    
        if local:
            feat.update({f"degree_{i}": degree[i] for i in range(len(degree))})
            feat.update({f"ec_{i}": ec[i] for i in range(len(ec))})

        else:
            feat["degree"] = degree.mean()
            feat["ec"] = ec.mean()


        if grafo_complementario:
            graph_c, Ac, Lc = gpl.genero_grafo_complementario(
                X[kk, :, :],
                threshold,
                range(19),
                channels_str,
                0,
            )

            ec_c, spec_ratio_c, spec_gap_c, le_c, degree_c, ac_c = gpl.GSA(graph_c, Ac, Lc)
            densidad_c, g_clus_c, l_clus_c = gpl.calculo_grafo(graph_c, [(11, 7), (12, 16)])
            entropy_c = gpl.calculo_entropia(graph_c, degree_c)

            feat_c = {
                "AC_c": abs(ac_c),
                "LE_c": le_c,
                "Entropy_c": entropy_c,
                "SR_c": spec_ratio_c,
                "SG_c": abs(spec_gap_c),
                "dens_c": densidad_c,
                "g_clus_c": g_clus_c,
            }

            if local:
                feat_c.update({f"degree_c_{i}": degree_c[i] for i in range(len(degree_c))})
                feat_c.update({f"ec_c_{i}": ec_c[i] for i in range(len(ec_c))})
            else:
                feat_c["degree_c"] = degree_c.mean()
                feat_c["ec_c"] = ec_c.mean()

            feat.update(feat_c)

        feature.append(feat)
    
    return feature


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
def umbral_global_train_dirigido(te_train, top_frac):

    vals_train = te_train.flatten()
    
    # Quitamos los ceros (la diagonal y las autoconexiones) para no sesgar el cuantil
    vals_train = vals_train[vals_train > 0]
    
    return np.quantile(vals_train, 1 - top_frac)



# %% [2] CARGA DE DATOS

########################## MAIN ###############################################

if __name__ == "__main__":
    
    print('#####################################################')
    print(' EVALUACIÓN 5-FOLD: TODAS LAS FEATURES')
    print('#####################################################')

    eeg_channels = ["Fz","Cz","Pz","C3","T3","C4","T4","Fp1","Fp2","F3","F4","F7","F8","P3","P4","T5","T6","O1","O2"]
    
    channel_map_e = {}
    for i, ch in enumerate(eeg_channels):
        channel_map_e[f"in_degree_{i}"] = f"in_d_{ch}"
        channel_map_e[f"out_degree_{i}"] = f"out_d_{ch}"
        channel_map_e[f"flow_{i}"] = f"flow_{ch}"
        channel_map_e[f"pagerank_{i}"] = f"pr_{ch}"
        channel_map_e[f"ec_mag_{i}"] = f"ec_mag_{ch}"
        channel_map_e[f"ec_fase_{i}"] = f"ec_fase_{ch}"
        channel_map_e[f"l_clus_{i}"] = f"l_clus_{ch}"
        
    ruta_base_carpetas = "../Entropia/TE_matrices_all_folds_train_validation_test_diag_zero"
    
    top_frac = 0.20
    local = True
    threshold_local = False
    q = 0.1

    # %% [3] EXTRACCIÓN DE MÉTRICAS    

    importancias_globales = []
    
    # Variables globales que usaremos en el Bloque 2
    memoria_folds = {}
    columnas_features = None

    y_reales_pacientes_global = []
    y_pred_pacientes_global = []
    y_proba_pacientes_global = []
    y_reales_ventanas_global = []
    y_pred_ventanas_global = []

    inicio = time.time()

    for fold in range(1, 6):
        print(f"--- Extrayendo y entrenando FOLD {fold} ---")
        # 1. Carga de datos
        X_train, y_train, ids_train = cargar_datos_split(ruta_base_carpetas, fold, "train")
        X_val, y_val, ids_val = cargar_datos_split(ruta_base_carpetas, fold, "validation")
        X_test, y_test, ids_test = cargar_datos_split(ruta_base_carpetas, fold, "test")
        
        # 2. Umbral
        threshold_e = umbral_global_train_dirigido(X_train, top_frac)
        # 3. Extracción de características
        df_train = pd.DataFrame(extraer_features_dirigidas(X_train, threshold_e, local, q, threshold_local, top_frac, eeg_channels))
        df_val   = pd.DataFrame(extraer_features_dirigidas(X_val, threshold_e, local, q, threshold_local, top_frac, eeg_channels))
        df_test  = pd.DataFrame(extraer_features_dirigidas(X_test, threshold_e, local, q, threshold_local, top_frac, eeg_channels))
        
        X_train_model = df_train.rename(columns=channel_map_e)
        X_val_model   = df_val.rename(columns=channel_map_e)
        X_test_model  = df_test.rename(columns=channel_map_e)
        
        columnas_features = X_train_model.columns # Guardamos los nombres de las columnas
        
        # Guardamos en memoria para reciclar en la etapa 2
        memoria_folds[fold] = {
            "X_train": X_train_model, "y_train": y_train,
            "X_val": X_val_model, "y_val": y_val,
            "X_test": X_test_model, "y_test": y_test, "ids_test": ids_test
        }
        
        # 4. Entrenamos el modelo
        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            early_stopping_rounds=20
        )
        model.fit(X_train_model, y_train, eval_set=[(X_val_model, y_val)], verbose=False)
        importancias_globales.append(model.feature_importances_)
        
        # 5. Predecimos
        predicciones_ventanas = model.predict(X_test_model)
        predicciones_proba = model.predict_proba(X_test_model)[:, 1]
        y_reales_ventanas_global.extend(y_test)
        y_pred_ventanas_global.extend(predicciones_ventanas)
        
        # Votación Mayoritaria Clínica
        pacientes_unicos = np.unique(ids_test)
        for paciente in pacientes_unicos:
            mascara_paciente = (ids_test == paciente)
            votos = predicciones_ventanas[mascara_paciente]
            votos_proba = predicciones_proba[mascara_paciente]
            etiqueta_real = y_test[mascara_paciente][0]
            
            voto_final = 1 if np.mean(votos) > 0.5 else 0
            proba_final = np.mean(votos_proba)
            
            y_reales_pacientes_global.append(etiqueta_real)
            y_pred_pacientes_global.append(voto_final)
            y_proba_pacientes_global.append(proba_final)

    # Calculamos la importancia promedio para usarla en el Bloque 2
    importancia_promedio = np.mean(importancias_globales, axis=0)

    # Resultados Finales (TODAS LAS FEATURES)
    print('\n' + '='*50)
    print(' RENDIMIENTO POR PACIENTE (TODAS LAS FEATURES)')
    print('='*50)
    print(f'Accuracy : {accuracy_score(y_reales_pacientes_global, y_pred_pacientes_global):.4f}')
    print(f'F1-Score : {f1_score(y_reales_pacientes_global, y_pred_pacientes_global):.4f}')
    print(f'Recall   : {recall_score(y_reales_pacientes_global, y_pred_pacientes_global):.4f}')
    print(f'Precision: {precision_score(y_reales_pacientes_global, y_pred_pacientes_global):.4f}')
    print(f'AUC ROC  : {roc_auc_score(y_reales_pacientes_global, y_proba_pacientes_global):.4f}')

    plt.figure(figsize=(12, 5))
    plt.bar(columnas_features, importancia_promedio)
    plt.ylabel("Importance")
    plt.title("Feature Importance Promedio (5-Folds) - Todas las features")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    fin = time.time()
    print(f"\nTiempo del Bloque 1: {fin - inicio:.2f} segundos")

    # %% [4] TOP N Features
    n_largest_feat = 6
    
    # 1. Obtenemos las Top N columnas usando el vector importancia_promedio del bloque anterior
    top_n_cols = pd.Series(importancia_promedio, index=columnas_features).nlargest(n_largest_feat).index
    
    print('\n' + '★'*50)
    print(f' RE-ENTRENANDO SOLO CON LAS TOP {n_largest_feat} FEATURES')
    print('★'*50)
    for idx, feature in enumerate(top_n_cols, 1):
        print(f"{idx}. {feature}")
    print('-'*50)

    y_reales_pacientes_top = []
    y_pred_pacientes_top = []
    y_proba_pacientes_top = []
    
    inicio_top = time.time()

    # 2. Iteramos usando los datos que ya teníamos en RAM
    for fold in range(1, 6):
        data = memoria_folds[fold]
        
        # Filtramos para quedarnos solo con las Top N columnas
        X_train_top = data["X_train"][top_n_cols]
        X_val_top   = data["X_val"][top_n_cols]
        X_test_top  = data["X_test"][top_n_cols]
        
        # Entrenamos de nuevo
        model_top = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            early_stopping_rounds=20
        )
        model_top.fit(X_train_top, data["y_train"], eval_set=[(X_val_top, data["y_val"])], verbose=False)
        
        # Predecimos
        predicciones_ventanas = model_top.predict(X_test_top)
        predicciones_proba = model_top.predict_proba(X_test_top)[:, 1]
        
        # Votación Mayoritaria Clínica
        pacientes_unicos = np.unique(data["ids_test"])
        for paciente in pacientes_unicos:
            mascara_paciente = (data["ids_test"] == paciente)
            votos = predicciones_ventanas[mascara_paciente]
            votos_proba = predicciones_proba[mascara_paciente]
            etiqueta_real = data["y_test"][mascara_paciente][0]
            
            voto_final = 1 if np.mean(votos) > 0.5 else 0
            proba_final = np.mean(votos_proba)
            
            y_reales_pacientes_top.append(etiqueta_real)
            y_pred_pacientes_top.append(voto_final)
            y_proba_pacientes_top.append(proba_final)

    # Resultados Finales Top N
    print('\n' + '='*50)
    print(f' RENDIMIENTO POR PACIENTE (TOP {n_largest_feat} FEATURES)')
    print('='*50)
    print(f'Accuracy : {accuracy_score(y_reales_pacientes_top, y_pred_pacientes_top):.4f}')
    print(f'F1-Score : {f1_score(y_reales_pacientes_top, y_pred_pacientes_top):.4f}')
    print(f'Recall   : {recall_score(y_reales_pacientes_top, y_pred_pacientes_top):.4f}')
    print(f'Precision: {precision_score(y_reales_pacientes_top, y_pred_pacientes_top):.4f}')
    print(f'AUC ROC  : {roc_auc_score(y_reales_pacientes_top, y_proba_pacientes_top):.4f}')

    # Gráfico de las Top N
    plt.figure(figsize=(8, 5))
    plt.bar(top_n_cols, pd.Series(importancia_promedio, index=columnas_features).nlargest(n_largest_feat))
    plt.ylabel("Importance")
    plt.title(f"Top {n_largest_feat} Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    fin_top = time.time()
    print(f"\nTiempo del Bloque 2: {fin_top - inicio_top:.2f} segundos")

# %%