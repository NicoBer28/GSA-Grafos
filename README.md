## 📋 Requisitos Previos

Es necesario tener instalado **Conda** (Anaconda o Miniconda).

---

## 📁 1. Descarga y Organización de Datos

Por razones de tamaño, las matrices de los datos no están almacenadas directamente en el repositorio.
Por eso, para poder correr los códigos, primero hay que seguir estos pasos:

1. Descargá la carpeta "data" desde el siguiente enlace: **[https://drive.google.com/drive/folders/1HiiU-7a-Y1WYslFE-X-CidD0q4e60hX2?usp=sharing]**
2. Descomprimí el contenido dentro de la raíz del proyecto **manteniendo exactamente la estructura original de carpetas**:

```text
GSA-Grafos/
├── data/
│   ├── EEG_crudo/
│   ├── Gaussian_Kernel/
│   └── ...
├── lib/
├── environment.yml
├── DGSAonTE.py
├── GSAonEEG.py
├── ...
└── README.md
```

***Importante***: No modifiques los nombres de los archivos ni la jerarquía de subcarpetas. Si ya tenes descargados los datos no hace falta seguir estos pasos, pero es importante mantener la jerarquía de subcarpetas y nombres de los archivos exactamente iguales.

## 🚀 2. Instalación y Ejecución
Abrí la terminal (o el Anaconda Prompt en Windows) dentro de la carpeta del proyecto y ejecutá los siguientes comandos:

### Paso 1: Crear el entorno virtual desde el archivo YAML

`conda env create -f environment.yml`

### Paso 2: Activar el entorno

`conda activate gsa`

### Paso 3: Ejecutar el modelo

`python DGSAonTE.py`    (o algún otro)

Para desactivar el entorno, correr:

`conda deactivate`