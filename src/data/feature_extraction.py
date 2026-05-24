"""
Módulo de Extracción de Características y Compresión Cuántica (Fase B).
Contiene todos los algoritmos desarrollados para reducir la dimensionalidad 
de 100 transiciones discretas a representaciones latentes o paramétricas.
"""

import numpy as np
import warnings
import time
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings('ignore')

# =====================================================================
# --- CLASE BASE (Plantilla Arquitectónica) ---
# =====================================================================
class BaseExtractor:
    """Clase padre que define la interfaz para todos los métodos de extracción."""
    def __init__(self, n_jobs=14):
        self.n_jobs = n_jobs

    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError("Cada submétodo debe implementar su propia lógica.")
    
# =====================================================================
# --- 1. MÉTODO: PCA CON NORMALIZACIÓN PROPORCIONAL ---
# =====================================================================
class PCAExtractor(BaseExtractor):
    """
    Técnica lineal (Baseline). Ajusta un PCA y escala los pesos (scores) 
    dividiéndolos por el máximo absoluto de la PC1.
    """
    def __init__(self, n_components=10, random_state=42, n_jobs=1):
        super().__init__(n_jobs)
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.factor_escala = 1.0

    def fit_transform(self, X_train):
        """
        Ajusta el PCA sobre los datos de entrenamiento y los normaliza.
        X_train: array-like de forma (n_muestras, n_features)
        """
        print(f"⚙️ Ajustando PCA ({self.n_components} componentes)...")
        pesos_raw = self.pca.fit_transform(X_train)
        
        # El cambio crítico: Normalización por el Máximo Absoluto de PC1
        self.factor_escala = np.max(np.abs(pesos_raw[:, 0]))
        print(f"✅ Factor de escala (Max Abs PC1): {self.factor_escala:.4f}")
        
        pesos_norm = pesos_raw / self.factor_escala
        return pesos_norm

    def transform(self, X_test):
        """Aplica la misma transformación y escala a datos de test."""
        pesos_raw = self.pca.transform(X_test)
        return pesos_raw / self.factor_escala
    
# =====================================================================
# --- 2. MÉTODO: LEAST SQUARES (Ajuste Híbrido Dinámico V1 y V2.5) ---
# =====================================================================
class LeastSquaresExtractor(BaseExtractor):
    """
    Motor Atómico Físico. 
    Ajusta 3 gaussianas ancla y N gaussianas libres (5 para el Modelo 8G, 7 para el 10G)
    distribuidas en zonas estancas con semillas optimizadas (Modelo 2).
    """
    def __init__(self, n_gaussianas=10, Omega_lambda=16.7, Omega_A=100.0, n_jobs=14):
        super().__init__(n_jobs)
        self.n_gaussianas = n_gaussianas
        self.Omega_lambda = Omega_lambda
        self.Omega_A = Omega_A

    @staticmethod
    def gaussiana(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

    # --- Funciones Matemáticas Dinámicas ---
    @staticmethod
    def suma_3_gaussianas(x, *p): return sum(LeastSquaresExtractor.gaussiana(x, p[i], p[i+1], p[i+2]) for i in range(0, 9, 3))
    
    @staticmethod
    def suma_5_gaussianas(x, *p): return sum(LeastSquaresExtractor.gaussiana(x, p[i], p[i+1], p[i+2]) for i in range(0, 15, 3))
    
    @staticmethod
    def suma_7_gaussianas(x, *p): return sum(LeastSquaresExtractor.gaussiana(x, p[i], p[i+1], p[i+2]) for i in range(0, 21, 3))

    @staticmethod
    def suma_8_gaussianas(x, *p): return sum(LeastSquaresExtractor.gaussiana(x, p[i], p[i+1], p[i+2]) for i in range(0, 24, 3))
    
    @staticmethod
    def suma_10_gaussianas(x, *p): return sum(LeastSquaresExtractor.gaussiana(x, p[i], p[i+1], p[i+2]) for i in range(0, 30, 3))

    @staticmethod
    def error_3(p, x, y_target): return LeastSquaresExtractor.suma_3_gaussianas(x, *p) - y_target
    @staticmethod
    def error_5(p, x, y_target): return LeastSquaresExtractor.suma_5_gaussianas(x, *p) - y_target
    @staticmethod
    def error_7(p, x, y_target): return LeastSquaresExtractor.suma_7_gaussianas(x, *p) - y_target
    @staticmethod
    def error_8(p, x, y_target): return LeastSquaresExtractor.suma_8_gaussianas(x, *p) - y_target
    @staticmethod
    def error_10(p, x, y_target): return LeastSquaresExtractor.suma_10_gaussianas(x, *p) - y_target

    def _procesar_molecula(self, args):
        idx, y_real, p9_inicial, wl_nm = args
        
        # --- CAPA 1: Ajuste de las 3 Principales (Anclaje) ---
        x0_3, b_lower_3, b_upper_3 = [], [], []
        for i in range(0, 9, 3):
            A, mu, sig = p9_inicial[i], p9_inicial[i+1], p9_inicial[i+2]
            margen_A = abs(A) * (self.Omega_A / 100.0)
            
            if A >= 0:
                A_l, A_u = max(2.0, A - margen_A), max(max(2.0, A - margen_A) + 0.1, A + margen_A)
            else:
                A_u, A_l = min(-2.0, A + margen_A), min(min(-2.0, A + margen_A) - 0.1, A - margen_A)
                
            A_start = np.clip(A, A_l, A_u)
            mu_l = np.clip(mu - self.Omega_lambda, 150.0, 649.9)
            mu_u = np.clip(mu + self.Omega_lambda, mu_l + 0.1, 650.0)
            mu_start = np.clip(mu, mu_l, mu_u)
            
            sig_l = np.clip(sig * 0.70, 2.0, 59.9)
            sig_u = np.clip(sig * 1.30, sig_l + 0.1, 60.0)
            sig_start = np.clip(sig, sig_l, sig_u)
            
            x0_3.extend([A_start, mu_start, sig_start])
            b_lower_3.extend([A_l, mu_l, sig_l])
            b_upper_3.extend([A_u, mu_u, sig_u])
            
        res_3 = least_squares(self.error_3, x0=x0_3, bounds=(b_lower_3, b_upper_3), args=(wl_nm, y_real), max_nfev=2500)
        popt_3 = res_3.x
            
        # --- CAPA 2: Matching Pursuit Estadístico (Zonas Dinámicas) ---
        y_parcial_3 = self.suma_3_gaussianas(wl_nm, *popt_3)
        y_residuo = y_real - y_parcial_3
        A_max_permitido = max([abs(popt_3[0]), abs(popt_3[3]), abs(popt_3[6])])
        
        # Selección de Modelo (Zonas y Funciones de Error)
        if self.n_gaussianas == 8:
            # MODELO 2: Semillas y Límites optimizados (5 zonas libres, sin solapamiento)
            zonas = [
                {"mu_start": 185.0, "mu_min": 150.0, "mu_max": 215.0},
                {"mu_start": 235.0, "mu_min": 215.1, "mu_max": 275.0},
                {"mu_start": 305.0, "mu_min": 275.1, "mu_max": 340.0},
                {"mu_start": 375.0, "mu_min": 340.1, "mu_max": 430.0},
                {"mu_start": 485.0, "mu_min": 430.1, "mu_max": 650.0}
            ]
            error_func_libre = self.error_5
            error_func_global = self.error_8
        else:
            # MODELO V2.5: 7 zonas libres para 10 gaussianas
            zonas = [
                {"mu_start": 177.4, "mu_min": 150.0, "mu_max": 182.9},
                {"mu_start": 195.2, "mu_min": 183.0, "mu_max": 211.9},
                {"mu_start": 225.0, "mu_min": 212.0, "mu_max": 239.9},
                {"mu_start": 270.0, "mu_min": 240.0, "mu_max": 289.9},
                {"mu_start": 308.0, "mu_min": 290.0, "mu_max": 324.9},
                {"mu_start": 335.0, "mu_min": 325.0, "mu_max": 369.9},
                {"mu_start": 405.0, "mu_min": 370.0, "mu_max": 650.0}
            ]
            error_func_libre = self.error_7
            error_func_global = self.error_10
        
        x0_libres, b_lower_libres, b_upper_libres = [], [], []
        for zona in zonas:
            mu_libre = zona["mu_start"]
            sigma_fisico = np.clip((mu_libre**2 / 1240.0) * 0.2, 2.1, 59.9)
            amp_inicial = np.clip(y_residuo[(np.abs(wl_nm - mu_libre)).argmin()], -A_max_permitido, A_max_permitido)
            sig_max = min(60.0, max(2.5, sigma_fisico * 3.0))
            
            x0_libres.extend([amp_inicial, mu_libre, sigma_fisico])
            b_lower_libres.extend([-A_max_permitido, zona["mu_min"], 2.0])
            b_upper_libres.extend([ A_max_permitido, zona["mu_max"], sig_max])
            
        res_libres = least_squares(error_func_libre, x0=x0_libres, bounds=(b_lower_libres, b_upper_libres), args=(wl_nm, y_residuo), max_nfev=2500)
        popt_libres = res_libres.x

        # --- CAPA 3: Ajuste Global ---
        x0_global = np.concatenate((popt_3, popt_libres))
        b_lower_global, b_upper_global = [], []
        for i in range(0, 9):
            val = popt_3[i]
            margen = abs(val) * 0.05
            b_lower_global.append(max(b_lower_3[i], val - margen))
            b_upper_global.append(min(b_upper_3[i], val + margen))
            
        b_lower_global.extend(b_lower_libres)
        b_upper_global.extend(b_upper_libres)
        
        res_global = least_squares(error_func_global, x0=x0_global, bounds=(b_lower_global, b_upper_global), args=(wl_nm, y_real), max_nfev=2500)
        
        # --- POST-PROCESAMIENTO: Zombis y Orden ---
        params_finales = list(res_global.x)
        num_params_totales = self.n_gaussianas * 3
        tripletas_libres = [params_finales[i:i+3] for i in range(9, num_params_totales, 3)]
        
        libres_procesadas = []
        for pico in tripletas_libres:
            if abs(pico[0]) < 1.5:  
                pico[0], pico[1], pico[2] = 0.0, 650.0, 2.0
            libres_procesadas.append(pico)
            
        libres_ordenadas = sorted(libres_procesadas, key=lambda pico: pico[1])
        return idx, np.concatenate((np.array(params_finales[0:9]), np.concatenate(libres_ordenadas)))

    def fit_transform(self, S_real_matrix, puros_9_params_matrix, wl_nm):
        N_mols = len(S_real_matrix)
        Y_params_final = np.zeros((N_mols, self.n_gaussianas * 3))
        
        print(f"🚀 Iniciando Least Squares (K={self.n_gaussianas}) en {self.n_jobs} núcleos ({N_mols} moléculas)...")
        tareas = [(idx, S_real_matrix[idx], puros_9_params_matrix[idx], wl_nm) for idx in range(N_mols)]
        
        resultados = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(self._procesar_molecula)(tarea) for tarea in tareas
        )
        
        for idx, params in resultados:
            Y_params_final[idx] = params
            
        print(f"✅ Ajuste Físico (K={self.n_gaussianas}) completado con éxito.")
        return Y_params_final
    # =====================================================================
# --- 3. MÉTODO: COMPRESIÓN CUÁNTICA (Clustering GMM / Agglomerative) ---
# =====================================================================
class QuantumClusteringExtractor(BaseExtractor):
    """
    Motor Cuántico: Transforma palitos discretos (transiciones) en distribuciones.
    Clona los puntos según su intensidad y aplica clustering probabilístico (GMM) 
    o jerárquico (Agglomerative) para encontrar Macro-Estados (semillas).
    """
    def __init__(self, algoritmo='gmm', k_clusters=5, n_jobs=14):
        super().__init__(n_jobs)
        self.algoritmo = algoritmo.lower()
        self.k_clusters = k_clusters
        self.hc_ev_nm = 1240.0
        self.sigma_base_ev = 0.2

    def _preparar_pool(self, wl, R, signo):
        """Filtra por signo y clona los puntos en función de su amplitud."""
        mask = (R > 0) if signo == 'pos' else (R < 0)
        wl_pool = wl[mask & ~np.isnan(wl)]
        R_pool = np.abs(R[mask & ~np.isnan(R)])
        
        if len(wl_pool) < self.k_clusters: 
            return None, 0.0
            
        factor = 2000 / (np.sum(R_pool) + 1e-9)
        puntos_clonados = []
        
        for w, r in zip(wl_pool, R_pool):
            num_clones = int(np.round(r * factor))
            puntos_clonados.extend([w] * num_clones)
            
        if len(puntos_clonados) < self.k_clusters: 
            return None, 0.0
            
        return np.array(puntos_clonados).reshape(-1, 1), np.sum(R_pool)

    def _calcular_sigma_instrumental(self, mu):
        return (mu**2 / self.hc_ev_nm) * self.sigma_base_ev + 1e-5

    def _procesar_molecula(self, args):
        """Procesa una molécula usando GMM o Agglomerative según configuración."""
        idx, wl_t, R_t = args
        params_pos, params_neg = [], []
        
        for signo in ['pos', 'neg']:
            X_cloned, sum_R = self._preparar_pool(wl_t, R_t, signo)
            
            # Párking de Zombis preventivo si no hay datos
            if X_cloned is None:
                zombis = [(0.0, 650.0, 2.0) for _ in range(self.k_clusters)]
                if signo == 'pos': params_pos = zombis.copy()
                else: params_neg = zombis.copy()
                continue
                
            temp_params = []
            
            if self.algoritmo == 'gmm':
                modelo = GaussianMixture(n_components=self.k_clusters, covariance_type='spherical', random_state=42, reg_covar=1e-3)
                modelo.fit(X_cloned)
                for k in range(self.k_clusters):
                    mu = modelo.means_[k][0]
                    sig_tot = np.sqrt(modelo.covariances_[k] + self._calcular_sigma_instrumental(mu)**2)
                    A = modelo.weights_[k] * sum_R
                    A = A if signo == 'pos' else -A
                    temp_params.append((A, mu, sig_tot))
                    
            elif self.algoritmo == 'agglomerative':
                modelo = AgglomerativeClustering(n_clusters=self.k_clusters)
                labels = modelo.fit_predict(X_cloned)
                for k in range(self.k_clusters):
                    puntos = X_cloned[labels == k].flatten()
                    mu = np.mean(puntos)
                    sig_clus = np.std(puntos) if len(puntos) > 1 else 0.0
                    sig_tot = np.sqrt(sig_clus**2 + self._calcular_sigma_instrumental(mu)**2)
                    A = (len(puntos) / len(X_cloned)) * sum_R
                    A = A if signo == 'pos' else -A
                    temp_params.append((A, mu, sig_tot))
            
            # Orden topológico
            temp_params.sort(key=lambda x: x[1])
            if signo == 'pos': params_pos = temp_params
            else: params_neg = temp_params

        # Aplanamos a un tensor 1D (30 parámetros)
        res = []
        for A, mu, sig in params_pos: res.extend([A, mu, sig])
        for A, mu, sig in params_neg: res.extend([A, mu, sig])
        
        return idx, np.array(res, dtype=np.float32)

    def fit_transform(self, wl_matrix, R_matrix):
        """Orquesta la extracción para toda la base de datos."""
        N_mols = len(wl_matrix)
        Y_params = np.zeros((N_mols, self.k_clusters * 6), dtype=np.float32)
        
        print(f"🚀 Iniciando Compresión Cuántica ({self.algoritmo.upper()}) en {self.n_jobs} núcleos...")
        tareas = [(i, wl_matrix[i], R_matrix[i]) for i in range(N_mols)]
        
        resultados = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(self._procesar_molecula)(tarea) for tarea in tareas
        )
        
        for idx, params in resultados:
            Y_params[idx] = params
            
        print(f"✅ ¡Base de datos {self.algoritmo.upper()} generada con éxito!")
        return Y_params