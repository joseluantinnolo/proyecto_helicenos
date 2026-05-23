"""
Módulo de Gestión de Datos, Normalización y Carga (Dataloaders).
Se encarga de ejecutar la extracción de características, guardar los archivos procesados
en el disco local (data/processed/) y empaquetar los datos en Tensores para PyTorch.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Importamos las herramientas que acabamos de programar
from src.data.feature_extraction import PCAExtractor, LeastSquaresExtractor, QuantumClusteringExtractor

class CDDatasetPipeline:
    """
    Pipeline maestro para la preparación de datos.
    Lee los datos crudos, extrae las características según el método elegido,
    guarda el resultado en disco para no perderlo y genera los DataLoaders.
    """
    def __init__(self, data_dir="../../data", batch_size=32, random_state=42):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Escaladores para la Red Neuronal
        self.scaler_X = StandardScaler()  # Para las posiciones Hammett
        self.scaler_Y = MinMaxScaler(feature_range=(-1, 1)) # Para los parámetros diana

        # Crear carpetas si no existen
        os.makedirs(self.processed_dir, exist_ok=True)

    def generar_y_guardar_parametros(self, metodo="gmm", **kwargs):
        """
        Ejecuta el extractor elegido, guarda los parámetros resultantes 
        en la carpeta data/processed/ para que no se pierdan, y los devuelve.
        """
        print(f"\n📦 Iniciando pipeline de extracción y guardado físico [Método: {metodo.upper()}]...")
        
        # 1. Carga básica y alineación de tus dos CSVs originales
        df_100 = pd.read_csv(os.path.join(self.raw_dir, 'Dataset_ECD_100R.csv'), sep=';')
        df_3 = pd.read_csv(os.path.join(self.raw_dir, 'DatasetDefinitivo_2310indep.csv'), sep=',')
        
        df_100 = df_100.drop_duplicates(subset=['Molecula']).reset_index(drop=True)
        df_3 = df_3.drop_duplicates(subset=['Molecula']).reset_index(drop=True)
        
        mols_comunes = set(df_100['Molecula']).intersection(set(df_3['Molecula']))
        mascara = df_100['Molecula'].isin(mols_comunes)
        
        # Extraer matrices cuánticas de 100 transiciones
        cols_wl = [f'nm_{i}' for i in range(1, 101)]
        cols_R = [f'R_{i}' for i in range(1, 101)]
        wl_matrix = df_100[cols_wl].values.astype(float)[mascara]
        R_matrix = df_100[cols_R].values.astype(float)[mascara]
        
        # Descriptores de entrada X (Las 16 posiciones de Hammett de las moléculas alineadas)
        cols_hammett = [c for c in df_3.columns if c.startswith('Pos_')]
        X_raw = df_3[cols_hammett].values.astype(float)[df_3['Molecula'].isin(mols_comunes)]
        
        # 2. Selección del Extractor según tu Plan de Discusión
        nombre_archivo_salida = f"Y_params_{metodo.lower()}_N10.npy"
        ruta_guardado = os.path.join(self.processed_dir, nombre_archivo_salida)
        
        # Inicializar variables
        Y_params = None
        
        if metodo.lower() in ['gmm', 'agglomerative']:
            extractor = QuantumClusteringExtractor(algoritmo=metodo, n_jobs=14)
            Y_params = extractor.fit_transform(wl_matrix, R_matrix)
            
        elif metodo.lower() == 'leastsquares':
            # Recuperamos los 9 parámetros crudos de anclaje para el Least Squares
            cols_ancla = ['Rmax', 'nm_Rmax', 'Rmin', 'nm_Rmin', 'R1', 'nm_R1'] # Ajustar orden según tu base
            # Como tu base tiene 6, rellenamos las 3 del tercer pico por compatibilidad
            puros_6p = df_3[cols_ancla].values.astype(float)[df_3['Molecula'].isin(mols_comunes)]
            puros_9p = np.zeros((len(puros_6p), 9))
            puros_9p[:, 0:6] = puros_6p # Copiamos los 6 parámetros
            puros_9p[:, 6] = puros_6p[:, 4] * 0.5 # Semilla de amplitud para la tercera
            puros_9p[:, 7] = 350.0                # Semilla de posición central
            puros_9p[:, 8] = 20.0                 # Semilla de anchura estándar
            
            # Malla universal
            wl_nm = np.linspace(150, 600, 100)
            
            # Simulamos tu matriz S_real
            from src.data.ground_truth_spectra import generar_envolvente_continua
            S_real_matrix = generar_envolvente_continua(wl_matrix, R_matrix, wl_nm)
            
            extractor = LeastSquaresExtractor(n_jobs=14)
            Y_params = extractor.fit_transform(S_real_matrix, puros_9p, wl_nm)
            
        elif metodo.lower() == 'pca':
            # El PCA se calcula sobre las envolventes finales de 100 puntos
            wl_nm = np.linspace(150, 600, 100)
            from src.data.ground_truth_spectra import generar_envolvente_continua
            S_real_matrix = generar_envolvente_continua(wl_matrix, R_matrix, wl_nm)
            
            extractor = PCAExtractor(n_components=kwargs.get('n_components', 10))
            Y_params = extractor.fit_transform(S_real_matrix)
            nombre_archivo_salida = f"Y_scores_pca_{extractor.n_components}.npy"
            ruta_guardado = os.path.join(self.processed_dir, nombre_archivo_salida)
            
        else:
            raise ValueError(f"Método '{metodo}' no reconocido para el pipeline.")
            
        # 3. GUARDADO FÍSICO EN DISCO (Aquí evitamos perder los datos)
        np.save(ruta_guardado, Y_params)
        np.save(os.path.join(self.processed_dir, "X_hammett_aligned.npy"), X_raw)
        
        print(f"💾 [OK] Datos guardados permanentemente en local:")
        print(f"   -> Entrada X: {os.path.join(self.processed_dir, 'X_hammett_aligned.npy')} | Forma: {X_raw.shape}")
        print(f"   -> Objetivo Y: {ruta_guardado} | Forma: {Y_params.shape}")
        
        return X_raw, Y_params

    def construir_loaders(self, X, Y, S_true=None, split_ratio=0.8):
        """
        Carga las matrices, aplica normalización rigurosa para PyTorch 
        y empaqueta todo en objetos DataLoader (Train/Val).
        """
        # 1. Normalización de variables (Ajuste sobre todo el espacio alineado)
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)
        
        if S_true is None:
            # Si no pasamos espectros reales (Caja Negra pura), creamos un lienzo vacío compatible
            S_true = np.zeros((len(X), 100))
            
        # 2. Split Manual determinista para reproducibilidad (Evita fugas de datos)
        N = len(X)
        split_idx = int(N * split_ratio)
        
        # Barajado síncrono mediante indexación
        indices = np.arange(N)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        idx_train, idx_val = indices[:split_idx], indices[split_idx:]
        
        # 3. Conversión a Tensores Diferenciables de PyTorch
        def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32)
        
        X_tr, X_va = to_tensor(X_scaled[idx_train]), to_tensor(X_scaled[idx_val])
        Y_tr, Y_va = to_tensor(Y_scaled[idx_train]), to_tensor(Y_scaled[idx_val])
        S_tr, S_va = to_tensor(S_true[idx_train]), to_tensor(S_true[idx_val])
        
        # 4. Creación de los DataLoaders estructurados (X, Y_parametros, Y_espectro)
        train_dataset = TensorDataset(X_tr, Y_tr, S_tr)
        val_dataset = TensorDataset(X_va, Y_va, S_va)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"\n📊 Tensores empaquetados con éxito para PyTorch:")
        print(f"   -> Lotes de entrenamiento (Train Loader): {len(train_loader)} batches de {self.batch_size}")
        print(f"   -> Lotes de validación (Val Loader): {len(val_loader)} batches")
        
        return train_loader, val_loader