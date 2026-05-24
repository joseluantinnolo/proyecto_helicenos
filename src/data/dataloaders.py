"""
Módulo de Gestión de Datos, Normalización y Carga (Dataloaders).
Integra el FILTRO FÍSICO y el ESCALADO COMPARTIDO para mantener la 
proporción real de las Gaussianas sin distorsionar los gradientes.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.data.feature_extraction import PCAExtractor, LeastSquaresExtractor, QuantumClusteringExtractor
# =====================================================================
# ESCALADOR FÍSICO PERSONALIZADO (UNIVERSAL)
# =====================================================================
class PhysicalScalerY:
    def __init__(self, y_dim):
        self.y_dim = y_dim
        self.is_3t = (y_dim == 6) # Fase A: 3 Transiciones (Sin Sigma)
        
        # Asignación dinámica de columnas (Pares para 3T, Tripletas para el resto)
        paso = 2 if self.is_3t else 3
        self.A_COLS = [i for i in range(0, y_dim, paso)]
        self.MU_COLS = [i for i in range(1, y_dim, paso)]
        
        if not self.is_3t:
            self.SIGMA_COLS = [i for i in range(2, y_dim, paso)]
        else:
            self.SIGMA_COLS = []
        
        # Constantes globales compartidas
        self.A_MAX = 100.0
        self.MU_MIN = 150.0
        self.MU_MAX = 650.0
        self.SIGMA_MAX = 60.0

    def fit(self, Y):
        self.A_MAX = float(np.max(np.abs(Y[:, self.A_COLS])))
        self.MU_MIN = float(np.min(Y[:, self.MU_COLS]))
        self.MU_MAX = float(np.max(Y[:, self.MU_COLS]))
        if not self.is_3t:
            self.SIGMA_MAX = float(np.max(Y[:, self.SIGMA_COLS]))
            
        print(f"📐 Constantes Físicas Extraídas: A_MAX={self.A_MAX:.2f} | RANGO_MU=[{self.MU_MIN:.2f}, {self.MU_MAX:.2f}]")

    def transform(self, Y):
        Y_norm = Y.copy()
        Y_norm[:, self.A_COLS] = Y[:, self.A_COLS] / self.A_MAX
        Y_norm[:, self.MU_COLS] = 2.0 * (Y[:, self.MU_COLS] - self.MU_MIN) / (self.MU_MAX - self.MU_MIN) - 1.0
        if not self.is_3t:
            Y_norm[:, self.SIGMA_COLS] = Y[:, self.SIGMA_COLS] / self.SIGMA_MAX
        return Y_norm.astype(np.float32)

    def fit_transform(self, Y):
        self.fit(Y)
        return self.transform(Y)

    def inverse_transform(self, Y_norm):
        Y = Y_norm.copy()
        Y[:, self.A_COLS] = Y_norm[:, self.A_COLS] * self.A_MAX
        Y[:, self.MU_COLS] = (Y_norm[:, self.MU_COLS] + 1.0) / 2.0 * (self.MU_MAX - self.MU_MIN) + self.MU_MIN
        if not self.is_3t:
            Y[:, self.SIGMA_COLS] = Y_norm[:, self.SIGMA_COLS] * self.SIGMA_MAX
        return Y.astype(np.float32)

# =====================================================================
# CLASE PRINCIPAL
# =====================================================================
class CDDatasetPipeline:
    def __init__(self, data_dir="../../data", batch_size=32, random_state=42):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.batch_size = batch_size
        self.random_state = random_state
        
        self.scaler_X = StandardScaler()  
        self.scaler_Y = None # Se inicializa dinámicamente según n_gaussianas

        os.makedirs(self.processed_dir, exist_ok=True)

    def generar_y_guardar_parametros(self, metodo="gmm", n_gaussianas=10, **kwargs):
        print(f"\n📦 Iniciando pipeline de extracción y guardado físico [Método: {metodo.upper()} | K={n_gaussianas}]...")
        
        nombre_archivo_salida = f"Y_params_{metodo.lower()}_N{n_gaussianas}.npy"
        ruta_guardado = os.path.join(self.processed_dir, nombre_archivo_salida)
        ruta_x = os.path.join(self.processed_dir, "X_hammett_aligned.npy")
        
        # =========================================================
        # 🚀 EL PARCHE SALVA-HORAS: Si ya existe, lo cargamos en 1 segundo
        # =========================================================
        if os.path.exists(ruta_guardado) and os.path.exists(ruta_x):
            print(f"⚡ [CACHE DETECTADA] ¡Datos encontrados en disco! Saltando extracción...")
            X_raw = np.load(ruta_x)
            Y_params = np.load(ruta_guardado)
            return X_raw, Y_params
        # =========================================================

        # 1. Carga básica y alineación de tus dos CSVs originales
        
        df_100 = pd.read_csv(os.path.join(self.raw_dir, 'Dataset_ECD_100R.csv'), sep=';')
        df_3 = pd.read_csv(os.path.join(self.raw_dir, 'DatasetDefinitivo_2310indep.csv'), sep=',')
        
        df_100 = df_100.drop_duplicates(subset=['Molecula']).reset_index(drop=True)
        df_3 = df_3.drop_duplicates(subset=['Molecula']).reset_index(drop=True)
        
        mols_comunes = set(df_100['Molecula']).intersection(set(df_3['Molecula']))
        mascara = df_100['Molecula'].isin(mols_comunes)
        
        cols_wl = [f'nm_{i}' for i in range(1, 101)]
        cols_R = [f'R_{i}' for i in range(1, 101)]
        wl_matrix = df_100[cols_wl].values.astype(float)[mascara]
        R_matrix = df_100[cols_R].values.astype(float)[mascara]
        
        cols_hammett = [c for c in df_3.columns if c.startswith('Pos_')]
        X_raw = df_3[cols_hammett].values.astype(float)[df_3['Molecula'].isin(mols_comunes)]
        
        nombre_archivo_salida = f"Y_params_{metodo.lower()}_N{n_gaussianas}.npy"
        ruta_guardado = os.path.join(self.processed_dir, nombre_archivo_salida)
        
        Y_params = None
        
        if metodo.lower() in ['gmm', 'agglomerative']:
            extractor = QuantumClusteringExtractor(algoritmo=metodo, k_clusters=n_gaussianas, n_jobs=14)
            Y_params = extractor.fit_transform(wl_matrix, R_matrix)
        elif metodo.lower() == 'leastsquares':
            cols_ancla = ['Rmax', 'nm_Rmax', 'Rmin', 'nm_Rmin', 'R1', 'nm_R1'] 
            puros_6p = df_3[cols_ancla].values.astype(float)[df_3['Molecula'].isin(mols_comunes)]
            puros_9p = np.zeros((len(puros_6p), 9))
            puros_9p[:, 0:6] = puros_6p 
            puros_9p[:, 6] = puros_6p[:, 4] * 0.5 
            puros_9p[:, 7] = 350.0                
            puros_9p[:, 8] = 20.0                 
            
            wl_nm = np.linspace(150, 600, 100)
            from src.data.ground_truth_spectra import generar_envolvente_continua
            S_real_matrix = generar_envolvente_continua(wl_matrix, R_matrix, wl_nm)
            
            extractor = LeastSquaresExtractor(n_gaussianas=n_gaussianas, n_jobs=14)
            Y_params = extractor.fit_transform(S_real_matrix, puros_9p, wl_nm)
        elif metodo.lower() == 'pca':
            wl_nm = np.linspace(150, 600, 100)
            from src.data.ground_truth_spectra import generar_envolvente_continua
            S_real_matrix = generar_envolvente_continua(wl_matrix, R_matrix, wl_nm)
            extractor = PCAExtractor(n_components=kwargs.get('n_components', 10))
            Y_params = extractor.fit_transform(S_real_matrix)
            ruta_guardado = os.path.join(self.processed_dir, f"Y_scores_pca_{extractor.n_components}.npy")
        else:
            raise ValueError(f"Método '{metodo}' no reconocido.")
            
        np.save(ruta_guardado, Y_params)
        np.save(os.path.join(self.processed_dir, "X_hammett_aligned.npy"), X_raw)
        
        return X_raw, Y_params

    def construir_loaders(self, X, Y, S_true=None, split_ratio=0.8):
        # 1. FILTRO FÍSICO UNIVERSAL
        if Y.shape[1] != 100: # Si NO es la Caja Negra (es paramétrico)
            is_3t = (Y.shape[1] == 6)
            paso = 2 if is_3t else 3
            
            MU_COLS = [i for i in range(1, Y.shape[1], paso)]
            mascara_validas = (np.max(Y[:, MU_COLS], axis=1) <= 650.0)
            
            if not is_3t:
                SIGMA_COLS = [i for i in range(2, Y.shape[1], paso)]
                mascara_validas &= (np.max(Y[:, SIGMA_COLS], axis=1) <= 60.0)
            
            n_antes = len(X)
            X = X[mascara_validas]
            Y = Y[mascara_validas]
            if S_true is not None: S_true = S_true[mascara_validas]
            print(f"🧹 Filtro de Calidad Físico: {n_antes - len(X)} moléculas descartadas. {len(X)} válidas para entrenar.")
            
            self.scaler_Y = PhysicalScalerY(y_dim=Y.shape[1])
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_Y = MinMaxScaler((-1, 1))

        # 2. Split Manual determinista (Test Size)
        N = len(X)
        split_idx = int(N * split_ratio)
        indices = np.arange(N)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        idx_train, idx_val = indices[:split_idx], indices[split_idx:]
        
        # 3. Normalización con Data Leakage Prevention
        X_tr_sc = self.scaler_X.fit_transform(X[idx_train])
        X_va_sc = self.scaler_X.transform(X[idx_val])
        
        self.scaler_Y.fit(Y[idx_train]) 
        Y_tr_sc = self.scaler_Y.transform(Y[idx_train])
        Y_va_sc = self.scaler_Y.transform(Y[idx_val])
        
        if S_true is None: S_true = np.zeros((len(X), 100))
        S_tr, S_va = S_true[idx_train], S_true[idx_val]
        
        # 4. Conversión a Tensores
        def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32)
        train_dataset = TensorDataset(to_tensor(X_tr_sc), to_tensor(Y_tr_sc), to_tensor(S_tr))
        val_dataset = TensorDataset(to_tensor(X_va_sc), to_tensor(Y_va_sc), to_tensor(S_va))
        
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)