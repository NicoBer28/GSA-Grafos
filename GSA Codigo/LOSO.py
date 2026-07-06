# # %%
# import os
# import numpy as np
# import scipy.io as sio

# # 1. Definimos las rutas a tus carpetas originales
# ruta_ctrl = "../EEG crudo/Control"
# ruta_tdah = "../EEG crudo/TDAH"

# # Listas maestras para acumular los datos
# todas_las_ventanas = []
# todos_los_ids = []
# todas_las_clases = []

# # Parámetros del ventaneo (Sliding Window) descubiertos
# tamaño_ventana = 512
# paso = 256  # 50% de overlap

# def procesar_carpeta_cruda(ruta, etiqueta_clase):
#     # Verificamos que la carpeta exista
#     if not os.path.exists(ruta):
#         print(f"Advertencia: No se encontró la ruta {ruta}")
#         return
        
#     for archivo in os.listdir(ruta):
#         if archivo.endswith(".mat"):
#             # Extraemos el ID exacto (ej: 'v113')
#             paciente_id = archivo.replace(".mat", "")
            
#             # Cargamos el archivo de MATLAB
#             ruta_completa = os.path.join(ruta, archivo)
#             mat_data = sio.loadmat(ruta_completa)
            
#             # Extraemos la señal usando el ID como llave secreta
#             # Si en algún archivo no coincide, usamos una búsqueda alternativa
#             if paciente_id in mat_data:
#                 eeg_crudo = mat_data[paciente_id]
#             else:
#                 # Busca la primera llave que no sea un metadato de MATLAB
#                 llaves_validas = [k for k in mat_data.keys() if not k.startswith('__')]
#                 eeg_crudo = mat_data[llaves_validas[0]]
            
#             # Sabemos que la forma es (tiempo, canales), ej: (15360, 19)
#             n_muestras_totales = eeg_crudo.shape[0]
            
#             # Aplicamos el Sliding Window con 50% de overlap
#             for inicio in range(0, n_muestras_totales - tamaño_ventana + 1, paso):
#                 fin = inicio + tamaño_ventana
                
#                 # Recortamos el bloque de tiempo
#                 ventana = eeg_crudo[inicio:fin, :]
                
#                 # Transponemos (.T) para que quede (19 canales, 512 muestras)
#                 ventana_acostada = ventana.T 
                
#                 # Guardamos la matriz y sus etiquetas correspondientes
#                 todas_las_ventanas.append(ventana_acostada)
#                 todos_los_ids.append(paciente_id)
#                 todas_las_clases.append(etiqueta_clase)
                
#     print(f"Carpeta '{etiqueta_clase}' procesada.")

# # 2. Ejecutamos la función para ambas clases
# print("Iniciando procesamiento de señales...")
# procesar_carpeta_cruda(ruta_ctrl, "Control")
# procesar_carpeta_cruda(ruta_tdah, "TDAH")

# # 3. Convertimos las listas maestras a tensores de NumPy
# X_in = np.array(todas_las_ventanas)
# subject_ids = np.array(todos_los_ids)
# class_names = np.array(todas_las_clases)

# # 4. Reporte de sanidad (Sanity Check)
# print("\n--- RESUMEN DEL NUEVO DATASET ---")
# print(f"Forma del tensor crudo (X_in): {X_in.shape}")
# print(f"Cantidad total de IDs: {len(subject_ids)}")
# print(f"Cantidad total de clases: {len(class_names)}")
# print("---------------------------------")

# # 5. Guardado del archivo unificado
# nombre_archivo_salida = "EEG_crudo_acotado.npz"
# np.savez(
#     nombre_archivo_salida, 
#     X_in=X_in, 
#     subject_ids=subject_ids, 
#     class_names=class_names
# )

# print(f"\n¡Éxito! Archivo guardado como: {nombre_archivo_salida}")

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

#-----LOSO-----
def cargar_datos_completos(ruta_npz, key_matrices="TE_matrices"):
    """
    Carga el archivo .npz unificado completo.
    key_matrices: "TE_matrices" para el nuevo modelo, "X_in" para el viejo.
    """
    print(f"Cargando archivo: {ruta_npz}")
    data = np.load(ruta_npz, allow_pickle=True)
    
    X = data[key_matrices]
    subject_ids = data["subject_ids"].astype(str)
    class_names = data["class_names"].astype(str)
    
    # Convertimos TDAH a 1 y Control a 0 para XGBoost
    y = (class_names == 'TDAH').astype(int)
    
    return X, y, subject_ids

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

def umbral_global_unificado(plv_matrices, y_labels, top_frac):
    # Separamos las matrices usando las etiquetas (0 = Control, 1 = TDAH)
    plv_ctrl = plv_matrices[y_labels == 0]
    plv_tdah = plv_matrices[y_labels == 1]
    
    n_channels = plv_ctrl.shape[1]
    iu = np.triu_indices(n_channels, k=1)

    vals_ctrl = plv_ctrl[:, iu[0], iu[1]].ravel()
    vals_tdah = plv_tdah[:, iu[0], iu[1]].ravel()
    vals_train = np.concatenate([vals_ctrl, vals_tdah])

    return np.quantile(vals_train, 1 - top_frac)


# %% [2] CARGA DE DATOS

########################## MAIN ###############################################

if __name__ == "__main__":
    
    print('#######################')
    print('EEG CRUDO')
    print('#######################')

    eeg_channels = ["Fz","Cz","Pz","C3","T3","C4","T4","Fp1","Fp2","F3","F4","F7","F8","P3","P4","T5","T6","O1","O2"]
    channel_map_e = {}
    channel_map = {}


    for i, ch in enumerate(eeg_channels):
        channel_map[f"degree_{i}"] = f"d_{ch}"
        channel_map[f"ec_{i}"] = f"ec_{ch}"
    
        channel_map_e[f"in_degree_{i}"] = f"in_d_{ch}"
        channel_map_e[f"out_degree_{i}"] = f"out_d_{ch}"
        channel_map_e[f"flow_{i}"] = f"flow_{ch}"
        channel_map_e[f"pagerank_{i}"] = f"pr_{ch}"
        channel_map_e[f"ec_mag_{i}"] = f"ec_mag_{ch}"
        channel_map_e[f"ec_fase_{i}"] = f"ec_fase_{ch}"
        channel_map_e[f"l_clus_{i}"] = f"l_clus_{ch}"
 
    
    # Cargo datos
    fs = 128
    n_chs = 19
    
    inicio = time.time()  # Captura el tiempo de inicio
    ruta_archivo_entropia = "../Entropia/TE_arrays_test_best_fold_2_diag_zero.npz"
    ruta_archivo_crudo = "../EEG crudo/EEG_crudo_acotado.npz"

    # %% [3] EXTRACCIÓN DE MÉTRICAS    
    threshold = 0.9
    threshold_e = 0.9   
    top_frac = 0.20
    local = True
    threshold_local = False
    grafo_complementario = False
    q=0.1
    # La variable q es un parámetro de "carga" que vos elegís
    # (suele ser un número chico como 0.1 o 0.25)
    # para darle más o menos importancia a la direccionalidad de las flechas.

    # %% LOSO
    X_crudo, y_completo, ids_completo = cargar_datos_completos(ruta_archivo_crudo, key_matrices="X_in")
    X_completo_e, y_completo_e, ids_completo_e = cargar_datos_completos(ruta_archivo_entropia, key_matrices="TE_matrices")

    n_ventanas = X_crudo.shape[0]
    plv_completo = np.zeros((n_ventanas, n_chs, n_chs))
    
    for i in range(n_ventanas):
        plv_completo[i,:,:] = phase_locking(X_crudo[i,:,:], fs)


    if threshold_local == False:
        threshold = umbral_global_unificado(plv_completo, y_completo, top_frac)
        print(f"Umbral global plv: {threshold:.4f}")

        threshold_e = umbral_global_train_dirigido(X_completo_e[y_completo_e==0], X_completo_e[y_completo_e==1], top_frac)
        print(f"Umbral global entropia: {threshold_e:.4f}")


    df_feat_completo = pd.DataFrame(extraer_features(
        plv_completo, threshold, local, threshold_local, grafo_complementario, top_frac, channels_str=eeg_channels
    ))
    X_features_modelo = df_feat_completo.rename(columns=channel_map)
    modelo_final, importancias = validacion_LOSO_por_paciente(X_features_modelo, y_completo, ids_completo)
    plt.figure(figsize=(10,5))
    plt.bar(X_features_modelo.columns, importancias)
    plt.ylabel("Importance")
    plt.title("Feature Importance (Modelo PLV - LOSO Completo)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    df_feat_completo_e = pd.DataFrame(extraer_features_dirigidas(
        X_completo_e, threshold_e, local, q, threshold_local, top_frac, channels_str=eeg_channels
    ))
    X_features_modelo_e = df_feat_completo_e.rename(columns=channel_map_e)
    modelo_final, importancias = validacion_LOSO_por_paciente(X_features_modelo_e, y_completo_e, ids_completo_e)
    plt.figure(figsize=(10,5))
    plt.bar(X_features_modelo_e.columns, importancias)
    plt.ylabel("Importance")
    plt.title("Feature Importance (Modelo Completo LOSO)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# %%
