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

class CDDatasetPipeline:
    def __init__(self, data_dir="../../data", batch_size=32, random_state=42):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.batch_size = batch_size
        self.random_state = random_state
        
        self.scaler_X = StandardScaler()  
        os.makedirs(self.processed_dir, exist_ok=True)

    def generar_y_guardar_parametros(self, metodo="gmm", n_gaussianas=10, **kwargs):
        print(f"\n📦 Iniciando pipeline de extracción y guardado físico [Método: {metodo.upper()} | K={n_gaussianas}]...")
        
        nombre_archivo_salida = f"Y_params_{metodo.lower()}_N{n_gaussianas}.npy"
        ruta_guardado = os.path.join(self.processed_dir, nombre_archivo_salida)
        ruta_x = os.path.join(self.processed_dir, "X_hammett_aligned.npy")
        
        # 🚀 EL PARCHE SALVA-HORAS
        if os.path.exists(ruta_guardado) and os.path.exists(ruta_x):
            print(f"⚡ [CACHE DETECTADA] ¡Datos encontrados en disco! Saltando extracción...")
            X_raw = np.load(ruta_x)
            Y_params = np.load(ruta_guardado)
            return X_raw, Y_params

        # 1. Carga básica y alineación
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
            
        elif metodo.lower() == 'caja_negra':
            wl_nm = np.linspace(150, 600, 100)
            from src.data.ground_truth_spectra import generar_envolvente_continua
            Y_params = generar_envolvente_continua(wl_matrix, R_matrix, wl_nm) # En Caja Negra, Y es el espectro
            ruta_guardado = os.path.join(self.processed_dir, "Y_espectros_cajanegra.npy")
            
        else:
            raise ValueError(f"Método '{metodo}' no reconocido.")
            
        np.save(ruta_guardado, Y_params)
        np.save(ruta_x, X_raw)
        return X_raw, Y_params

    def get_physical_passport(self, Y, metodo):
        """Genera el diccionario de escalas físicas globales para la función de pérdida."""
        passport = {'metodo': metodo.lower()}
        
        if metodo.lower() in ['gmm', 'leastsquares', 'agglomerative']: 
            y_dim = Y.shape[1]
            is_3t = (y_dim == 6)
            paso = 2 if is_3t else 3
            
            A_cols = [1, 3, 5] if is_3t else [i for i in range(0, y_dim, paso)]
            MU_cols = [0, 2, 4] if is_3t else [i for i in range(1, y_dim, paso)]
            
            passport['a_max'] = float(np.max(np.abs(Y[:, A_cols])))
            passport['mu_min'] = float(np.min(Y[:, MU_cols]))
            passport['mu_max'] = float(np.max(Y[:, MU_cols]))
            
            if not is_3t:
                SIGMA_cols = [i for i in range(2, y_dim, paso)]
                passport['sigma_max'] = float(np.max(Y[:, SIGMA_cols]))
                
        elif metodo.lower() in ['pca', 'caja_negra']:
            passport['a_max'] = float(np.max(np.abs(Y)))
            
        return passport

    def construir_loaders(self, X, Y, S_true, metodo, split_ratio=0.8):
        # 1. FILTRO FÍSICO (Solo para PINNs)
        if metodo.lower() not in ['pca', 'caja_negra']:
            is_3t = (Y.shape[1] == 6)
            paso = 2 if is_3t else 3
            MU_COLS = [0, 2, 4] if is_3t else [i for i in range(1, Y.shape[1], paso)]
            mascara = (np.max(Y[:, MU_COLS], axis=1) <= 650.0)
            
            if not is_3t:
                SIGMA_COLS = [i for i in range(2, Y.shape[1], paso)]
                mascara &= (np.max(Y[:, SIGMA_COLS], axis=1) <= 60.0)
            
            n_antes = len(X)
            X, Y = X[mascara], Y[mascara]
            if S_true is not None: S_true = S_true[mascara]
            print(f"🧹 Filtro Físico: {n_antes - len(X)} moléculas descartadas. {len(X)} válidas.")

        # 2. Generar el Pasaporte con los datos limpios
        passport = self.get_physical_passport(Y, metodo)

        # 3. Normalizar Y (y S_true) usando el pasaporte
        Y_norm = Y.copy()
        if metodo.lower() in ['gmm', 'leastsquares', 'agglomerative']:
            is_3t = (Y.shape[1] == 6)
            paso = 2 if is_3t else 3
            A_cols = [1, 3, 5] if is_3t else [i for i in range(0, Y.shape[1], paso)]
            MU_cols = [0, 2, 4] if is_3t else [i for i in range(1, Y.shape[1], paso)]
            
            Y_norm[:, A_cols] = Y[:, A_cols] / passport['a_max']
            Y_norm[:, MU_cols] = 2.0 * (Y[:, MU_cols] - passport['mu_min']) / (passport['mu_max'] - passport['mu_min']) - 1.0
            
            if not is_3t:
                SIGMA_cols = [i for i in range(2, Y.shape[1], paso)]
                Y_norm[:, SIGMA_cols] = Y[:, SIGMA_cols] / passport['sigma_max']
                
        elif metodo.lower() in ['pca', 'caja_negra']:
            Y_norm = Y / passport['a_max']

        # S_true siempre se escala con a_max para que Loss tenga sentido
        if S_true is not None:
            S_norm = S_true / passport['a_max']
        else:
            S_norm = np.zeros((len(X), 100))

        # 4. Split y Normalización de X
        N = len(X)
        split_idx = int(N * split_ratio)
        indices = np.arange(N)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        idx_train, idx_val = indices[:split_idx], indices[split_idx:]

        X_tr_sc = self.scaler_X.fit_transform(X[idx_train])
        X_va_sc = self.scaler_X.transform(X[idx_val])

        # 5. Tensores
        def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32)
        train_ds = TensorDataset(to_tensor(X_tr_sc), to_tensor(Y_norm[idx_train]), to_tensor(S_norm[idx_train]))
        val_ds = TensorDataset(to_tensor(X_va_sc), to_tensor(Y_norm[idx_val]), to_tensor(S_norm[idx_val]))
        
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=True), DataLoader(val_ds, batch_size=self.batch_size, shuffle=False), passport