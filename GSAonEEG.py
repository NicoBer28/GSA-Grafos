# %%  [1] LIBRERIAS Y FUNCIONES

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os
import lib.GSA_lib as gsa
import time
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score



def leodatos(path):
    train_tdah = np.load(path)
    #print(train_tdah.files)
    X = train_tdah['X_in']
    #print(X.shape)
    return X


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
        graph, A, L = gsa.genero_grafo(
            X[kk, :, :], 
            threshold, 
            range(19), 
            channels_str,
            0
        )
        ec, spec_ratio, spec_gap, le, degree, ac = gsa.GSA(graph, A, L)
        densidad, g_clus, l_clus = gsa.calculo_grafo(graph, [(11, 7), (12, 16)])
        entropy = gsa.calculo_entropia(graph, degree)


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
            graph_c, Ac, Lc = gsa.genero_grafo_complementario(
                X[kk, :, :],
                threshold,
                range(19),
                channels_str,
                0,
            )

            ec_c, spec_ratio_c, spec_gap_c, le_c, degree_c, ac_c = gsa.GSA(graph_c, Ac, Lc)
            densidad_c, g_clus_c, l_clus_c = gsa.calculo_grafo(graph_c, [(11, 7), (12, 16)])
            entropy_c = gsa.calculo_entropia(graph_c, degree_c)

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
    
    print('#####################################################')
    print('EEG CRUDO')
    print('#####################################################')
    print('\n')

    eeg_channels = ["Fz","Cz","Pz","C3","T3","C4","T4","Fp1","Fp2","F3","F4","F7","F8","P3","P4","T5","T6","O1","O2"]
    channel_map = {}

    bandas = {
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta":  (13.0, 30.0),
        "Gamma":  (30.0, 100.0)
    }
    filtrar_por_banda = False

    for i, ch in enumerate(eeg_channels):
        channel_map[f"degree_{i}"] = f"d_{ch}"
        channel_map[f"ec_{i}"] = f"ec_{ch}"

        # Para las métricas del grafo complementario
        channel_map[f"degree_c_{i}"] = f"d_c_{ch}"
        channel_map[f"ec_c_{i}"] = f"ec_c_{ch}"
 
    
    # Cargo datos
    fs = 128
    n_chs = 19
    
    inicio = time.time()  # Captura el tiempo de inicio

    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_base_carpetas = os.path.normpath(os.path.join(directorio_script, "./data/EEG_crudo"))

    X_ctrl_train = leodatos(os.path.normpath(os.path.join(ruta_base_carpetas, 'train_class0_fold2.npz')))
    X_ctrl_test = leodatos(os.path.normpath(os.path.join(ruta_base_carpetas, 'test_class0_fold2.npz')))
    X_tdah_train = leodatos(os.path.normpath(os.path.join(ruta_base_carpetas, 'train_class1_fold2.npz')))
    X_tdah_test = leodatos(os.path.normpath(os.path.join(ruta_base_carpetas, 'test_class1_fold2.npz')))
    # Inicializar las métricas con listas vacias

    if filtrar_por_banda: 
        nombre_banda = "Beta"
        f_min, f_max = bandas[nombre_banda]
        X_ctrl_train = filtrar_banda_eeg(X_ctrl_train, f_min, f_max, fs)
        X_ctrl_test = filtrar_banda_eeg(X_ctrl_test, f_min, f_max, fs)
        X_tdah_train = filtrar_banda_eeg(X_tdah_train, f_min, f_max, fs)
        X_tdah_test = filtrar_banda_eeg(X_tdah_test, f_min, f_max, fs)

    print('#####################################################')
    print(' EVALUACIÓN TODAS LAS FEATURES...')
    print('#####################################################')

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

    #threshold = 0.9   
    top_frac = 0.20
    local = True
    threshold_local = False
    grafo_complementario = False
    #print(threshold)
    if threshold_local == False:
        threshold = umbral_global_train(plv_ctrl_train, plv_tdah_train, top_frac)
        print(f"Umbral global calculado: {threshold:.4f}")

    ctrl_feat_train  = pd.DataFrame(extraer_features(plv_ctrl_train , threshold, local, threshold_local, grafo_complementario, top_frac, channels_str=eeg_channels))
    tdah_feat_train = pd.DataFrame(extraer_features(plv_tdah_train, threshold, local, threshold_local, grafo_complementario, top_frac, channels_str=eeg_channels))
    ctrl_feat_test  = pd.DataFrame(extraer_features(plv_ctrl_test , threshold, local, threshold_local, grafo_complementario, top_frac, channels_str=eeg_channels))
    tdah_feat_test = pd.DataFrame(extraer_features(plv_tdah_test, threshold, local, threshold_local, grafo_complementario, top_frac, channels_str=eeg_channels))


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

    model, metricas, importances = clasifico_Xgboost(X_train_model,y_train)

    # Evaluo el modelo con todas las features en datos de TEST
    y_todas_pred = model.predict(X_test_model)
    acc_test = accuracy_score(y_test, y_todas_pred)
    f1_test = f1_score(y_test, y_todas_pred)
    recall_test = recall_score(y_test, y_todas_pred)
    precision_test = precision_score(y_test, y_todas_pred)
    auc_test = roc_auc_score(y_test, y_todas_pred)
    
    print('\n' + '='*50)
    print(' RENDIMIENTO (TODAS LAS FEATURES)')
    print('='*50)
    print(f'Accuracy: {acc_test:.4f}')
    print(f'F1: {f1_test:.4f}')
    print(f'Recall: {recall_test:.4f}')
    print(f'Precision: {precision_test:.4f}')
    print(f'AUC ROC  : {auc_test:.4f}')

    inicio2 = time.time()

    plt.figure()
    plt.bar(X_train_model.columns, importances)
    plt.ylabel("Importance")
    plt.title("Raw EEG")
    plt.show()
    plt.figure()

    fin2 = time.time()
    # %% [5] ENTRENAMIENTO CON TOP N FEATURES
    n_largest_feat = 6
    top_n_cols = pd.Series(importances, index=X_train_model.columns).nlargest(n_largest_feat).index

    print('\n' + '★'*50)
    print(f' RE-ENTRENANDO SOLO CON LAS TOP {n_largest_feat} FEATURES')
    print('★'*50)
    for idx, feature in enumerate(top_n_cols, 1):
        print(f"{idx}. {feature}")
    print('-'*50)
    
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
    
    print('\n' + '='*50)
    print(f' RENDIMIENTO (TOP {n_largest_feat} FEATURES)')
    print('='*50)
    print(f'Accuracy: {acc_test:.4f}')
    print(f'F1: {f1_test:.4f}')
    print(f'Recall: {recall_test:.4f}')
    print(f'Precision: {precision_test:.4f}')
    if auc_test is not None:
        print(f'AUC ROC: {auc_test:.4f}')
    

    fin = time.time()

    plt.bar(X_top_n_test.columns, importances)
    plt.ylabel("Importance")
    plt.title("Raw EEG")
    plt.show()

    print(f"Tiempo de ejecución: {fin - inicio - fin2 + inicio2:.2f} segundos")

    
# %%
