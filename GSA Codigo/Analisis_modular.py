#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 09:56:06 2025

@author: mariapau
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:40:50 2024

@author: mariapau
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import Grafos_Paula_lib as gpl
import scipy as sp
import scikit_posthocs as spost
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import FormatStrFormatter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier  # Para clasificación
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats
from scipy.stats import chi2_contingency

###########################################################################################
# Función para crear gráficos de barras de error
def grafico_errorbar(serie_A, serie_C, serie_F, metrica):
    
    """Genera un gráfico de barras de error para las tres series dadas.
    Serie_A es una lista de 5 items (thresholds), en cada item hay 35*4 epocas para A, 
    30*4 epocas para C, 22*4 epocas para F
    """
    escalas = {'AC' : 21, 'LE': 90, 'Density': 1.2, 'Entropy': 2.5, 'SR': 19, 
               'SG': 1.2, 'g_clust': 19, 'EC_prom': 0.25, 'Distance': 4, 'l_clust': 1}
    titulos = {'AC' : 'AC', 'LE': 'LE', 'Density': 'Density', 'Entropy': 'Entropy', 'SR': 'SR', 'l_clust': 'Local \ Cluster',
               'SG': 'SG', 'g_clust': 'Glob \ Clust', 'EC_prom': 'av \ EC', 'Distance': 'Path \ Length'}
    z,p0 = sp.stats.kruskal(serie_A[0], serie_C[0], serie_F[0])
    z,p1 = sp.stats.kruskal(serie_A[1], serie_C[1], serie_F[1])
    z,p2 = sp.stats.kruskal(serie_A[2], serie_C[2], serie_F[2])
    z,p3 = sp.stats.kruskal(serie_A[3], serie_C[3], serie_F[3])
    z,p4 = sp.stats.kruskal(serie_A[4], serie_C[4], serie_F[4])
    print(metrica, ': ', p0, p1, p2, p3, p4)
    try: 
        dunn_resultados = spost.posthoc_dunn([serie_A[0], serie_C[0], serie_F[0]], p_adjust='bonferroni')  # Corrección de Bonferroni
        dunn_resultados = spost.posthoc_dunn([serie_A[1], serie_C[1], serie_F[1]], p_adjust='bonferroni')  # Corrección de Bonferroni
        dunn_resultados = spost.posthoc_dunn([serie_A[2], serie_C[2], serie_F[2]], p_adjust='bonferroni')  # Corrección de Bonferroni
        dunn_resultados = spost.posthoc_dunn([serie_A[3], serie_C[3], serie_F[3]], p_adjust='bonferroni')  # Corrección de Bonferroni
        dunn_resultados = spost.posthoc_dunn([serie_A[4], serie_C[4], serie_F[4]], p_adjust='bonferroni')  # Corrección de Bonferroni
        print('Dunn Resultados')
        print(dunn_resultados)    
    except:
        print('Error')
      

    std1 = np.nanstd(serie_A, axis = 1)
    std2 = np.nanstd(serie_C, axis = 1)
    std3 = np.nanstd(serie_F, axis = 1)
    m1 = np.nanmean(serie_A, axis = 1)
    m2 = np.nanmean(serie_C, axis = 1)
    m3 = np.nanmean(serie_F, axis = 1)
    
    fig, ax = plt.subplots()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticks(range(5))
    ax.set_xticklabels([0.5, 0.6, 0.7, 0.8, 0.9],  fontsize=20)
    #ax.set_yticklabels(ax.get_yticks(), fontsize=15)
    ax.tick_params(axis='y', labelsize=20) 
    ax.set_xlabel('Thresholds',  fontsize=24)
    ax.errorbar(range(5), m1, yerr=std1, label=" $AD$", linewidth = 2)
    ax.errorbar(range(5), m2, yerr=std2, label=" $C$", linewidth = 2)
    ax.errorbar(range(5), m3, yerr=std2, label=" $FTD$", linewidth = 2)
    ax.set_title(f"$ {titulos[metrica]} \ ({ritmo})$", fontsize=24)
    ax.legend(fontsize=16)
    fig.subplots_adjust(bottom=0.15)  # Aumentar margen inferior

    
    amp = max(max(m1),max(m2),max(m3)) + max(max(std1), max(std2), max(std3))
    amp = escalas[metrica]
    
    P = np.array([p0, p1, p2, p3, p4])
    ax.plot(range(5), 0.94*amp*(np.where(P<0.05, 1, np.nan)), 'k*')

    # Guardar el topomap como archivo
    output_file = "/Users/mariapau/Conicet/Grafos_IAM /Codigo/" + ritmo + '_' + metrica + ".pdf"
    fig.savefig(output_file, dpi=400)  # Guardar como PNG con alta resolución


###########################################################################################
# Regresion globales y mmse y edad


def regresion_mmse(x,y, label):
    
    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    
    # Predicciones
    y_pred = model.predict(x.reshape(-1, 1))
    
    # Cálculo de métricas
    r2 = model.score(x.reshape(-1, 1), y)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Mostrar resultados
    print(f"R^2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Graficar datos y recta de regresión
    plt.figure()
    plt.scatter(x, y, label="Datos reales", color="blue")
    plt.plot(x, y_pred, label="Ajuste lineal", color="red")
    plt.xlabel(f'Variable {label}')
    plt.ylabel("Variable MMSE")
    plt.title(f'Ajuste Lineal MMSE vs {label}')
    plt.legend()
    plt.grid()
    plt.show()
    
###########################################################################################
# Ejemplo de uso modular
# Agregar lógica aquí para cargar datos y procesarlos modularmente.
# Utilizar las funciones y estructuras creadas para organizar el análisis.

def grafico_kde(grupos, umb, ritmo, r, metricas_claves, age):
    #Armo matriz de caracteristicas totales
    labelsss = ['dens','$AC$', '$LE$', '$SG$', '$SR$', '$plength$','$EC_{av}$', '$Age$']
    c = 7
    X = np.zeros((260,c+1))
    # MÉTRICAS PARA VALIDACIÓN
    scoring = {
        'accuracy': 'accuracy',
        'recall': 'recall',
        'f1': 'f1'
    }

    
    fig, axs = plt.subplots(c,1, figsize=(3,15))


    for l,param in enumerate(metricas_claves[1:c+1]):
    #for l,param in enumerate(['LE', 'EC_prom', 'SG', 'SR']):
        axs[l].set_ylabel(labelsss[l+1], fontsize= 16)
        A = grupos['A'][param][umb]
        C = grupos['C'][param][umb]
        sns.kdeplot(A, label="AD", color="blue", ax = axs[l])
        sns.kdeplot(C, label="C", color="orange", ax = axs[l])
        axs[l].set_yticklabels([])
        axs[l].set_xticklabels(axs[l].get_xticks(), fontsize=14)
        # Formatear yticks a un decimal
        axs[0].legend()
        AC = np.concatenate((A,C), axis = 0)
        X[:,l] = AC

    X[:,l+1] = age
    #X[:,l+2] = sex
    
    plt.subplots_adjust(hspace=0.7)  # Espaciado vertical entre subgráficos
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Guardar el topomap como archivo
    output_file = "/Users/mariapau/Conicet/Grafos_IAM /Codigo/kde_" + ritmo + ".pdf"
    fig.savefig(output_file, dpi=400)  # Guardar como PNG con alta resolución

    # ARMO MODELO RANDOM FOREST
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
    
    y = [0]*140 + [1]*120
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=r)
    #model = RandomForestClassifier(n_estimators=100, random_state=r)
    model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    # VALIDACIÓN CRUZADA
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    
    # IMPRIMIR RESULTADOS
    print('#############################################################################')
    print("Accuracy (por pliegue):", cv_results['test_accuracy'])
    print("Recall (por pliegue):", cv_results['test_recall'])
    print("F1-score (por pliegue):", cv_results['test_f1'])
    
    # Promedio y desviación estándar de cada métrica
    print(f"Mean accuracy: {cv_results['test_accuracy'].mean():.2f} ± {cv_results['test_accuracy'].std():.2f}")
    print(f"Mean recall: {cv_results['test_recall'].mean():.2f} ± {cv_results['test_recall'].std():.2f}")
    print(f"Mean F1-score: {cv_results['test_f1'].mean():.2f} ± {cv_results['test_f1'].std():.2f}")

    plt.figure()
    plt.imshow(np.corrcoef(X.T))
    model.fit(X_train, y_train)
    # Obtener la importancia de características
    importancias = model.feature_importances_
    
    # Asociar los valores con los nombres de las características
    feat = metricas_claves[1:c+1]
    feat.append('age')
    #feat.append('sex')
    feature_importance_df = pd.DataFrame({
        'Feature': feat,
        'Importance': importancias
    }).sort_values(by='Importance', ascending=False)    
    print(feature_importance_df)
    
    # Obtener las probabilidades de predicción para la clase positiva
    y_scores = model.predict_proba(X_test)[:, 1]
     
     # Calcular la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
     # Graficar la curva ROC
    plt.figure(17,figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Global features (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for GSA features, {ritmo}, threshold: {thresholds[umb]}')
    plt.legend(loc='lower right')
    plt.show()
    return roc_auc, fpr, tpr
    
###########################################################################################
def grafico_barras(grupos, claves):
    equis = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    ancho = 0.3

    for clave in claves:
        fig, ax = plt.subplots(5, 1, figsize=(4.5, 10))  # Relación de aspecto más vertical con mayor altura
        plt.suptitle(clave, y= 0.92)
        for n in range(5):
            aux = grupos['A'][clave][n]
            prom_A = [np.mean(aux[:7]), np.mean(aux[7:10]),np.mean(aux[10:14]),np.mean(aux[14:17]),np.mean(aux[17:])]
            stdd_A = [np.std(aux[:7]), np.std(aux[7:10]),np.std(aux[10:14]),np.std(aux[14:17]),np.std(aux[17:])]
            aux = grupos['C'][clave][n]
            prom_C = [np.mean(aux[:7]), np.mean(aux[7:10]),np.mean(aux[10:14]),np.mean(aux[14:17]),np.mean(aux[17:])]
            stdd_C = [np.std(aux[:7]), np.std(aux[7:10]),np.std(aux[10:14]),np.std(aux[14:17]),np.std(aux[17:])]
            # Gráfico (puedes descomentar y ajustar si es necesario)
            # grafico_por_umbral(EC_A_th[n], EC_C_th[n], EC_F_th[n], n, 'EC by threshold', thresholds)
            ax[n].bar(equis, prom_A, label='AD', width=0.5, yerr=stdd_A, capsize=5)
            ax[n].bar(equis+ancho, prom_C, label='C', width=0.5, yerr=stdd_C, capsize=5)
            # Define las posiciones y etiquetas del eje x
            ax[n].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
            ax[n].set_xticklabels(['F', 'C', 'T', 'P', 'O'])
            # Fija el rango del eje x
            ax[n].set_xlim(0, 5)
            # Fija el rango del eje y
            #ax[n].set_ylim([0, 1.1])
            ax[n].legend()
            ax[n].set_ylabel(f'Threshold = {thresholds[n]}')
    # Realizar el test de comparacion de dos distribuciones no normales
    stat, p_value = sp.stats.ranksums(grupos['A'][clave][n], grupos['C'][clave][n])
            
    # Resultados
    print('###########################################################')
    print(clave)
    print(f"n = {n}, Estadístico: {stat}")
    print(f"n = {n}, p-valor: {p_value}")
    plt.show()

    # Guardar el topomap como archivo
    output_file = "/Users/mariapau/Conicet/Grafos_IAM /Codigo/barras_" + clave + '_' + ritmo + ".pdf"
    fig.savefig(output_file, dpi=300)  # Guardar como PNG con alta resolución


###########################################################################################
def clasifico(grupos, umb, r, claves, age):
    
    X_ = []
    # MÉTRICAS PARA VALIDACIÓN
    scoring = {
        'accuracy': 'accuracy',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    for clave in claves:
        aux = grupos['A'][clave][umb]
        prom_A = np.array([np.mean(aux[:,:7],axis = 1), np.mean(aux[:,7:10], axis = 1),np.mean(aux[:,10:14],  axis = 1),np.mean(aux[:,14:17],  axis = 1),np.mean(aux[:,17:],  axis = 1)])
        aux = grupos['C'][clave][umb]
        prom_C = np.array([np.mean(aux[:,:7],axis = 1), np.mean(aux[:,7:10], axis = 1),np.mean(aux[:,10:14],  axis = 1),np.mean(aux[:,14:17],  axis = 1),np.mean(aux[:,17:],  axis = 1)])
        X_.append(np.concatenate((prom_A.T, prom_C.T), axis = 0))
   
    #X = np.concatenate((X_[0], X_[1], age[:,np.newaxis]), age[:,np.newaxis], axis = 1)  # concateno las features de DC y EC
    X = np.concatenate((X_[0], X_[1], age[:,np.newaxis]), axis = 1)  # concateno las features de DC y EC
    #X = np.concatenate((X_[0], X_[1]), axis = 1)  # concateno las features de DC y EC
    #X = X[:,[7,1,3,2]]
    # ARMO MODELO RANDOM FOREST
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
    
    y = [0]*140 + [1]*120
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=r)

    #model = RandomForestClassifier(n_estimators=100, random_state=r)
    model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 300, 500],
        'subsample': [0.7, 1.0]
    }
    
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)

    print("Mejores hiperparámetros:", grid_search.best_params_)
    
    
    # VALIDACIÓN CRUZADA
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', learning_rate = 0.01, max_depth = 3, n_estimators = 100, subsample = 1.0)
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    
    # IMPRIMIR RESULTADOS
    print('#############################################################################')
    print("Accuracy (por pliegue):", cv_results['test_accuracy'])
    print("Recall (por pliegue):", cv_results['test_recall'])
    print("F1-score (por pliegue):", cv_results['test_f1'])
    
    # Promedio y desviación estándar de cada métrica
    print(f"Mean accuracy: {cv_results['test_accuracy'].mean():.2f} ± {cv_results['test_accuracy'].std():.2f}")
    print(f"Mean recall: {cv_results['test_recall'].mean():.2f} ± {cv_results['test_recall'].std():.2f}")
    print(f"Mean F1-score: {cv_results['test_f1'].mean():.2f} ± {cv_results['test_f1'].std():.2f}")

    model.fit(X_train, y_train)
    # Obtener la importancia de características
    importancias = model.feature_importances_

    # Asociar los valores con los nombres de las características
    feature_importance_df = pd.DataFrame({
        #'Feature': ['EC_F', 'EC_C', 'EC_T', 'EC_P', 'EC_O', 'D_F', 'D_C', 'D_T', 'D_P', 'D_O', 'age','sex'],
        'Feature': ['EC_F', 'EC_C', 'EC_T', 'EC_P', 'EC_O', 'D_F', 'D_C', 'D_T', 'D_P', 'D_O', 'age'],
        #'Feature': ['EC_F', 'EC_C', 'EC_T', 'EC_P', 'EC_O', 'D_F', 'D_C', 'D_T', 'D_P', 'D_O'],
        'Importance': importancias
    }).sort_values(by='Importance', ascending=False)    
    print(feature_importance_df)
    
   # Obtener las probabilidades de predicción para la clase positiva
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Calcular la curva ROC
    fpr_r, tpr_r, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr_r, tpr_r)
    
    # Graficar la curva ROC
    plt.figure(17, figsize=(8, 6))
    plt.plot(fpr_r, tpr_r, 'red', lw=2, label=f'ROC Regional features (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve for global vs regional GSA features')
    plt.legend(loc='lower right')
    plt.show()
    return roc_auc, fpr_r, tpr_r

#############################################################################
###################     MAIN                        #########################
#############################################################################
# Estructura de datos para agrupar métricas
grupos = {"A": {}, "C": {}, "F": {}}
metricas_claves = ["Density", "AC", "LE", "SG", "SR", "Distance", "EC_prom", "g_clust",  "l_clust", "EC_reg", "l_clust_raw", "EC", "Degree", 'disconnected', "Power", "entr_sig", "Entropy"]

# Cargar archivo TSV con los datos demográficos
parent_dir = '/Users/mariapau/ds004504-download/derivatives'
demog_df = pd.read_csv(parent_dir[:-12] + "/participants.tsv", sep="\t")
mmse = demog_df['MMSE'].to_numpy()
age = demog_df['Age'].to_numpy()
sex = demog_df['Gender'].to_numpy()
sex = np.where(sex=='F', 0, 1)

age_ = np.tile(age[:, np.newaxis], (1, 4)).T  # Repite el vector en 5 columnas
mmse_ = np.tile(mmse[:, np.newaxis], (1, 4)).T  # Repite el vector en 5 columnas
sex_ = np.tile(sex[:, np.newaxis], (1, 4)).T  # Repite el vector en 5 columnas

# age_col = age_.reshape(-1, order='F')[:-23*4]   #elimino los F
# mmse_col = mmse_.ravel(order = 'F')[:-23*4]   #elimino los F
# sex_col = sex_.reshape(-1, order='F')[:-23*4]   #elimino los F

age_col = age_.reshape(-1, order='F')   #elimino los F
mmse_col = mmse_.ravel(order = 'F')   #elimino los F
sex_col = sex_.reshape(-1, order='F')   #elimino los F


age_col = np.delete(age_col, np.arange(65*4,88*4))
mmse_col = np.delete(mmse_col, np.arange(65*4,88*4))
sex_col = np.delete(sex_col, np.arange(65*4,88*4))

# Inicializar las métricas
for grupo in grupos:
    for clave in metricas_claves:
        grupos[grupo][clave] = [[] for _ in range(5)]


if __name__ == "__main__":
    
    # Constantes y datos iniciales
    pares = [(11,7),(12,16)]
    ritmo = 'theta'
    plt.close('all')

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    #thresholds = [0.5]

    Pts_idx = [[0,35],[35,65],[65,87]]
    #Pts_idx = [[30,35],[62,64],[82,84]]

    ##############################################################################
      
    for n,th in enumerate(thresholds):            
        
        AC, dist, grado, l_clus, g_clus, entr, densidad, LE, SG, SR, EC, pot, entr_sig =  gpl.leo_archivos(ritmo, th, Pts_idx)    
        for grupo in grupos:
            idx = np.where(dist[grupo]>1e4)
            dist[grupo][idx] = np.nan
            grupos[grupo]['disconnected'][n] = len(idx[0])
            
        for grupo in grupos:
            grupos[grupo]['AC'][n] = abs(AC[grupo]).flatten()
            grupos[grupo]['LE'][n] = LE[grupo].flatten()
            grupos[grupo]['g_clust'][n] = abs(g_clus[grupo].flatten())
            grupos[grupo]['Entropy'][n] = np.array(entr[grupo])
            grupos[grupo]['entr_sig'][n] = np.array(entr_sig[grupo])
            grupos[grupo]['SG'][n] = abs(SG[grupo].flatten())
            grupos[grupo]['SR'][n] = abs(SR[grupo].flatten())
            grupos[grupo]['Density'][n] = densidad[grupo]
            grupos[grupo]['l_clust'][n] = np.nanmean(l_clus[grupo].reshape(len(densidad[grupo]),19), axis = 1)
            grupos[grupo]['EC_prom'][n] = np.nanmean(abs(EC[grupo]).reshape(len(densidad[grupo]),19), axis = 1)
            grupos[grupo]['EC_reg'][n] = np.nanmean(abs(EC[grupo]).reshape(len(densidad[grupo]),19), axis = 0)
            grupos[grupo]['EC'][n] = abs(EC[grupo]).reshape(len(densidad[grupo]),19)
            grupos[grupo]['l_clust_raw'][n] = l_clus[grupo].reshape(len(densidad[grupo]),19)
            grupos[grupo]['Degree'][n] = grado[grupo].reshape(len(densidad[grupo]),19)
            grupos[grupo]['Distance'][n] = np.nanmean(dist[grupo].reshape(len(densidad[grupo]),2), axis = 1)
            grupos[grupo]['Power'] = pot[grupo]

    # Placeholder para integración de lógica principal
    print("Archivo modular cargado correctamente.")
    umb = 2 
    r = 38
    for param in metricas_claves[:-8]:
        grafico_errorbar(grupos["A"][param], grupos["C"][param], grupos["F"][param], param)
        
    auc_g_gamma, fpr_g_gamma, tpr_g_gamma = grafico_kde(grupos, umb, ritmo, r, metricas_claves[:-8], age_col)    
    #auc_g_gamma, fpr_g_gamma, tpr_g_gamma = grafico_kde(grupos, umb, ritmo, r, metricas_claves[:-8], age_col, sex_col)    
    #grafico_kde(grupos,  umb, ritmo, r, metricas_claves[:-8],0)    
        
    grafico_barras(grupos, ['EC', 'Degree', 'l_clust'])
    
    auc_r_gamma, fpr_r_gamma, tpr_r_gamma = clasifico(grupos, umb, r, ['EC', 'Degree'], age_col)
    #auc_r_gamma, fpr_r_gamma, tpr_r_gamma = clasifico(grupos, umb, r, ['EC', 'Degree'], age_col, sex_col)
    #clasifico(grupos,  umb, r, ['EC', 'Degree'],0)

    
     ##############################################################################
     #Ploteo desconectados
     
    fig, ax = plt.subplots()
    for grupo in grupos:
         ax.plot(np.array(grupos[grupo]['disconnected'])/348, linewidth = 2, label = grupo)
         ax.set_xticks(range(5))
         ax.set_xticklabels([0.5, 0.6, 0.7, 0.8, 0.9],  fontsize=18)
         #ax.set_yticklabels(ax.get_yticks(), fontsize=15)
         ax.tick_params(axis='y', labelsize=18) 
         ax.set_xlabel('Thresholds',  fontsize=22)
         ax.set_title(f"$ Disconnected \ ({ritmo})$", fontsize=24)
         ax.legend(fontsize=16)
         fig.subplots_adjust(bottom=0.15)  # Aumentar margen inferior
         ax.set_ylim(0,0.9)
             
         output_file = "/Users/mariapau/Conicet/Grafos_IAM /Codigo/disconnected_" + ritmo + ".pdf"
         plt.savefig(output_file, dpi=300)  # Guardar como PNG con alta resolución

    ##############################################################################
    #Testeo confounders
    
        # Filtrar los grupos
    A = demog_df[demog_df['Group'] == 'A'].copy()
    C = demog_df[demog_df['Group'] == 'C'].copy()
    F = demog_df[demog_df['Group'] == 'F'].copy()
    
     # Prueba de Mann-Whitney U
    print('')
    print('Mann-whitney Edad: ')
    stat, p_value = stats.mannwhitneyu(A['Age'], C['Age'], alternative='two-sided')
    print(f'Av women age: {np.mean(A["Age"])} +- {np.std(A["Age"])}, av men age: {np.mean(C["Age"])}+- {np.std(C["Age"])}')
    # Resultados
    print(f'Estadístico U: {stat}')
    print(f'Valor p: {p_value}')
    

    # Mapear Gender a valores numéricos (1 = F, 0 = M)
    A['Gender_num'] = A['Gender'].map({'F': 1, 'M': 0})
    C['Gender_num'] = C['Gender'].map({'F': 1, 'M': 0})
    
    # Unir en un solo DataFrame para la tabla de contingencia
    group_labels = ['A'] * len(A) + ['C'] * len(C)
    gender_labels = list(A['Gender_num']) + list(C['Gender_num'])
    
    # Crear tabla de contingencia
    table = pd.crosstab(pd.Series(group_labels, name='Group'),
                        pd.Series(gender_labels, name='Gender'))
    
    print("Tabla de contingencia:\n", table)
    
    # Chi-cuadrado
    chi2, p_chi2, _, _ = chi2_contingency(table)
    print("Chi-squared test p-value:", p_chi2)

        
    # Interpretación
    alpha = 0.05
    if p_value < alpha:
        print("Rechazamos la hipótesis nula: hay diferencias significativas entre los grupos.")
    else:
        print("No se rechaza la hipótesis nula: no hay evidencia de diferencias significativas")

    print(f"AD MMSE: {np.mean(A['MMSE'])} +- {np.std(A['MMSE'])} ")
    print(f"C MMSE: {np.mean(C['MMSE'])} +- {np.std(C['MMSE'])} ")
    print(f"F MMSE: {np.mean(F['MMSE'])} +- {np.std(F['MMSE'])} ")
   
    """
    ##############################################################################
    #Testeo la regresion de los features con MMSE
    mmse = demog_df['MMSE'].to_numpy()
    mmse_col = mmse_.ravel(order = 'F')   
    #elimino los C
    mmse_col = np.delete(mmse_col, np.arange(35*4,66*4))
    
    for k in metricas_claves[:7]:
        caract = np.concatenate([grupos[clave][k][0] for clave in ['A', 'F']])
        try:
            regresion_mmse(caract, mmse_col, k)
        except:
            print('no se pudo')
            
         
          
    # Graficar la curva ROC globales
    plt.figure()
    plt.plot(fpr_g_delta, tpr_g_delta, lw=2, label=f'Delta, (AUC = {auc_g_delta:.2f})')
    plt.plot(fpr_g_theta, tpr_g_theta,  lw=2, label=f'Theta, (AUC = {auc_g_theta:.2f})')
    plt.plot(fpr_g_beta, tpr_g_beta, lw=2, label=f'Beta, (AUC = {auc_g_beta:.2f})')
    plt.plot(fpr_g_alpha, tpr_g_alpha, lw=2, label=f'Alpha, (AUC = {auc_g_alpha:.2f})')
    plt.plot(fpr_g_gamma, tpr_g_gamma,  lw=2, label=f'Gamma, (AUC = {auc_g_gamma:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for global GSA features', fontsize=14)
    plt.legend(loc='lower right', fontsize=13)
    plt.show()
    
    # Graficar la curva ROC regionales
    plt.figure()
    plt.plot(fpr_r_delta, tpr_r_delta, lw=2, label=f'Delta, (AUC = {auc_r_delta:.2f})')
    plt.plot(fpr_r_theta, tpr_r_theta,  lw=2, label=f'Theta, (AUC = {auc_r_theta:.2f})')
    plt.plot(fpr_r_beta, tpr_r_beta, lw=2, label=f'Beta, (AUC = {auc_r_beta:.2f})')
    plt.plot(fpr_r_alpha, tpr_r_alpha, lw=2, label=f'Alpha, (AUC = {auc_r_alpha:.2f})')
    plt.plot(fpr_r_gamma, tpr_r_gamma,  lw=2, label=f'Gamma, (AUC = {auc_r_gamma:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for regional GSA features', fontsize=14)
    plt.legend(loc='lower right', fontsize=13)
    plt.show()

    """
                