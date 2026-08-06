# %%  [1] LIBRERIAS Y FUNCIONES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import lib.DGSA_lib as dgsa
import time
from xgboost import XGBClassifier
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


def extraer_features_dirigidas(
    X,
    threshold,
    local,
    q,
    top_frac=0.5,
    channels_str=None,
):
    feature = []
    
    for kk in range(X.shape[0]):

        graph, A, L = dgsa.genero_grafo_dirigido(
            X[kk, :, :], 
            threshold, 
            range(X.shape[1]),
            channels_str,
            q,
            ploteo=0,
        )
        
        ac, spec_gap, ec_magnitud, ec_fase = dgsa.DGSA(L)
        densidad, g_clus, l_clus, reciprocidad, in_degree, out_degree, flujo, pr = dgsa.calculo_grafo_dirigido(graph, A)
        

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
    print(' TE')
    print('#####################################################')
    print('\n')

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
    
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_base_carpetas = os.path.normpath(os.path.join(directorio_script, "./data/TE"))
    
    top_frac = 0.20
    local = True
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

    print('#####################################################')
    print(' EVALUACIÓN TODAS LAS FEATURES')
    print('#####################################################')

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
        df_train = pd.DataFrame(extraer_features_dirigidas(X_train, threshold_e, local, q, top_frac, eeg_channels))
        df_val   = pd.DataFrame(extraer_features_dirigidas(X_val, threshold_e, local, q, top_frac, eeg_channels))
        df_test  = pd.DataFrame(extraer_features_dirigidas(X_test, threshold_e, local, q, top_frac, eeg_channels))
        
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
    print(' RENDIMIENTO (TODAS LAS FEATURES)')
    print('='*50)
    print(f'Accuracy : {accuracy_score(y_reales_pacientes_global, y_pred_pacientes_global):.4f}')
    print(f'F1-Score : {f1_score(y_reales_pacientes_global, y_pred_pacientes_global):.4f}')
    print(f'Recall   : {recall_score(y_reales_pacientes_global, y_pred_pacientes_global):.4f}')
    print(f'Precision: {precision_score(y_reales_pacientes_global, y_pred_pacientes_global):.4f}')
    print(f'AUC ROC  : {roc_auc_score(y_reales_pacientes_global, y_proba_pacientes_global):.4f}')

    fin = time.time()

    plt.figure(figsize=(12, 5))
    plt.bar(columnas_features, importancia_promedio)
    plt.ylabel("Importance")
    plt.title("Feature Importance Promedio (5-Folds) - Todas las features")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

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
    print(f' RENDIMIENTO (TOP {n_largest_feat} FEATURES)')
    print('='*50)
    print(f'Accuracy : {accuracy_score(y_reales_pacientes_top, y_pred_pacientes_top):.4f}')
    print(f'F1-Score : {f1_score(y_reales_pacientes_top, y_pred_pacientes_top):.4f}')
    print(f'Recall   : {recall_score(y_reales_pacientes_top, y_pred_pacientes_top):.4f}')
    print(f'Precision: {precision_score(y_reales_pacientes_top, y_pred_pacientes_top):.4f}')
    print(f'AUC ROC  : {roc_auc_score(y_reales_pacientes_top, y_proba_pacientes_top):.4f}')

    fin_top = time.time()

    # Gráfico de las Top N
    plt.figure(figsize=(8, 5))
    plt.bar(top_n_cols, pd.Series(importancia_promedio, index=columnas_features).nlargest(n_largest_feat))
    plt.ylabel("Importance")
    plt.title(f"Top {n_largest_feat} Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"\nTiempo del Bloque 2: {fin_top - inicio_top:.2f} segundos")

# %%