#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:55:35 2025

@author: mariapau
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import mne
import pickle
import Grafos_Paula_lib as gpl
import time


def compute_coherence(data, sfreq, fmin, fmax):
    """
    Calcula la coherencia entre todos los pares de canales EEG.

    Parámetros:
    - data: array (n_canales, n_muestras) con señales EEG
    - sfreq: frecuencia de muestreo en Hz
    - fmin, fmax: banda de frecuencia de interés (ejemplo: 8-12 Hz para alfa)

    Retorna:
    - coherence_matrix: matriz de coherencia (n_canales, n_canales)
    """

    n_channels, _ = data.shape
    coherence_matrix = np.zeros((n_channels, n_channels))

    # Calcular coherencia para cada par de canales
    for i in range(n_channels):
        for j in range(i+1, n_channels):  # Solo calcular triangulo superior
            f, Cxy = signal.coherence(data[i, :], data[j, :], fs=sfreq, nperseg=256)

            # Filtrar la coherencia en la banda de interés
            band_coherence = np.mean(Cxy[(f >= fmin) & (f <= fmax)])

            coherence_matrix[i, j] = band_coherence
            coherence_matrix[j, i] = band_coherence  # Matriz simétrica

    return coherence_matrix


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


def extract_epochs(raw, epoch_duration=7, max_epochs=4, start_offset=500):
    """
    Extracts fixed-duration epochs from an EEG recording, ensuring no overlap with 'boundary' annotations.
    """
    sfreq = raw.info['sfreq']
    epoch_samples = int(epoch_duration * sfreq)
    boundary_samples = np.array([int(annot['onset'] * sfreq) for annot in raw.annotations if annot['description'] == 'boundary'])

    epochs = []
    start_sample = start_offset

    while start_sample + epoch_samples <= len(raw.times) and len(epochs) < max_epochs:
        end_sample = start_sample + epoch_samples
        if not any(start_sample < b < end_sample for b in boundary_samples):
            epoch = raw.copy().crop(tmin=start_sample / sfreq, tmax=end_sample / sfreq, include_tmax=False)
            epochs.append(epoch.get_data())
        start_sample += epoch_samples

    return epochs

def read_and_segment_eeg(directory):
    """Reads .set files and extracts valid EEG epochs from subfolders."""
    all_epochs = []
    for subfolder in sorted(os.listdir(directory)):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.startswith("sub-"):
            eeg_path = os.path.join(subfolder_path, "eeg")
            if os.path.isdir(eeg_path):
                for file in os.listdir(eeg_path):
                    if file.endswith(".set"):
                        try:
                            raw = mne.io.read_raw_eeglab(os.path.join(eeg_path, file), preload=True)
                            all_epochs.extend(extract_epochs(raw))
                        except Exception as e:
                            print(f"Error processing {file}: {e}")

    print(f"Extracted {len(all_epochs)} EEG epochs in total.")
    return all_epochs

def filter_signal(data, fs, band):
    """Filters EEG data for a specific frequency band."""
    b, a = signal.butter(2, band, fs=fs, btype='bandpass')
    return signal.filtfilt(b, a, data)

def calculate_entropy_signal(signal_data):
    """Calculates entropy of a signal using its histogram."""
    probabilities, _ = np.histogram(signal_data, bins=20, density=True)
    probabilities = probabilities[probabilities > 0]
    return stats.entropy(probabilities, base=2)

def process_epochs(epochs, fs, channels, band_range, threshold, pares):
    """Processes EEG epochs to calculate graph metrics and signal properties."""
    
    for epoch_data in epochs:
            data_freq_total = filter_signal(epoch_data, fs, [0.5, 45])
            filtered_data = filter_signal(epoch_data, fs, band_range)
            can = 9
            power_rel = np.sum(np.abs(np.fft.fft(filtered_data[can,:])))/ np.sum(np.abs(np.fft.fft(data_freq_total[can,:])))
            entropy_sig = calculate_entropy_signal(filtered_data)        
            #mat_cov = np.corrcoef(filtered_data)
            mat_cov = phase_locking(filtered_data, fs)
            #mat_cov = compute_coherence(epoch_data, fs, rhythm_bands[ritmo[0]][0], rhythm_bands[ritmo[0]][1])
            graph, A, L = gpl.genero_grafo(mat_cov, threshold, channels, [f'Ch{i}' for i in range(len(channels))], 0)
            ec, spec_ratio, spec_gap, le, degree, ac = gpl.GSA(graph, A, L)
            densidad, g_clus, l_clus = gpl.calculo_grafo(graph, [(11, 7), (12, 16)])
            dist = gpl.calc_distancias(graph, pares)
            entropy = gpl.calculo_entropia(graph, degree)
            results["AC"].append(ac)
            results["dist_T4_T3_T5_T6"].append(dist)
            results["density"].append(densidad)
            #results["entropy"].append(1)
            results["entropy"].append(entropy)
            #results["entropy_sig"].append(1)
            results["entropy_sig"].append(entropy_sig)
            results["SR"].append(spec_ratio)
            results["SG"].append(spec_gap)
            results["global_clustering"].append(ac)
            results["local_clustering"].append(l_clus)
            results["degree_distribution"].append(degree)
            results["EC"].append(ec)
            results["LE"].append(le)
            results["power"].append(power_rel)
            #results["power"].append(1)


    return results

def save_results(results, output_dir):
    """Saves processed results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    for key, value in results.items():
        with open(os.path.join(output_dir, f"{key}.pkl"), "wb") as f:
            pickle.dump(pd.DataFrame(value), f)


########################## MAIN ###############################################

if __name__ == "__main__":
    # Recorre los 5 umbrales, y los 5 ritmos
    inicio = time.time()  # Captura el tiempo de inicio
       
    # Estructura de datos para agrupar métricas
    pares = [(11,7),(12,16)]
    grupos = {"A": {}, "C": {}, "F": {}}
    metricas_claves = ["Density", "AC", "LE", "Entropy", "g_clust", "SG", "SR", "Distance", "l_clust", "EC", "Degree", "entropy_sig", "Power", "MMSE", "age"]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    # Inicializar las métricas con listas vacias
    for grupo in grupos:
        for clave in metricas_claves:
            grupos[grupo][clave] = [[] for _ in range(5)]

    output = '/Users/mariapau/Conicet/Grafos_IAM '
    parent_dir = '/Users/mariapau/ds004504-download/derivatives'
    fs = 500
    rhythm_bands = {
        "delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]
    }

    # Cargar archivo TSV con los datos demográficos
    # demog_df = pd.read_csv(parent_dir[:-12] + "/participants.tsv", sep="\t")
    # results["MMSE"].append(demog_df['MMSE'].to_numpy())
    # results["age"].append(demog_df["Age"].to_numpy())
    # save_results(results, output)

    epochs = read_and_segment_eeg(parent_dir)
    for ritmo in rhythm_bands.items():
        for th in thresholds:
            print('Ritmo: ', ritmo, 'Umbral: ', th)
            
            results = {
            "density": [], "entropy": [], "LE": [], "SR": [], "SG": [], "AC": [], "dist_T4_T3_T5_T6": [],
            "local_clustering": [], "global_clustering": [], "EC": [], "degree_distribution": [], "entropy_sig": [], "power": []
            }
            
            results = process_epochs(epochs, fs, list(range(19)), ritmo[1], th, pares)
            output_dir = output + '/' + ritmo[0] + '/' + str(th) + '/' 
            save_results(results, output_dir)

    fin = time.time()  # Captura el tiempo de inicio
    print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")

