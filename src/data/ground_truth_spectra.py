"""
Módulo de Preprocesamiento Físico-Matemático para Dicroísmo Circular.
Responsable de generar envolventes continuas a partir de transiciones discretas
mediante la aproximación de campanas de Gauss.

Arquitectura: Completamente vectorizada y agnóstica a la dimensionalidad 
(soporta N transiciones y M moléculas simultáneamente).
"""

import os
import warnings
import numpy as np
import pandas as pd

# =====================================================================
# --- CONSTANTES FÍSICAS GLOBALES ---
# =====================================================================
# Factor de conversión entre Energía (eV) y Longitud de onda (nm)
# E(eV) = hc / lambda(nm) -> hc = 1240.0 eV*nm
HC_EV_NM: float = 1240.0 

# Ancho de banda de la campana de Gauss por defecto (en eV)
DEFAULT_SIGMA_EV: float = 0.2 

# Término de estabilidad numérica para evitar divisiones por cero
EPSILON: float = 1e-5 

# =====================================================================
# --- 1. MOTOR DE CÁLCULO VECTORIZADO ---
# =====================================================================
def generar_envolvente_continua(
    wl_transitions: np.ndarray, 
    r_transitions: np.ndarray, 
    wl_grid: np.ndarray, 
    sigma_ev: float = DEFAULT_SIGMA_EV
) -> np.ndarray:
    """
    Genera el espectro continuo de dicroísmo circular a partir de transiciones discretas.
    Agnóstica a la dimensión (soporta 3 o 100 transiciones de forma nativa).
    """
    # 1. Ajuste de dimensiones para el Broadcasting (Inyección de ejes)
    wl_t_3d = wl_transitions[..., np.newaxis]
    r_t_3d = r_transitions[..., np.newaxis]
    
    # Aseguramos que la malla sea unidimensional
    wl_grid_1d = np.asarray(wl_grid).flatten()
    
    # 2. Física: Cálculo de sigma variable (de eV a nm)
    sigma_nm_3d = ((wl_t_3d**2) / HC_EV_NM) * sigma_ev + EPSILON
    
    # Ignoramos alertas matemáticas por NaNs temporales
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 3. Núcleo Matemático Gaussiano vectorizado
        exponente = -((wl_grid_1d - wl_t_3d)**2) / (2 * (sigma_nm_3d**2))
        matriz_gaussianas = r_t_3d * np.exp(exponente)
        
        # 4. Compresión sumando las gaussianas
        espectro_continuo = np.nansum(matriz_gaussianas, axis=-2)
        
    return espectro_continuo

# =====================================================================
# --- 2. INTEGRACIÓN CON PANDAS ---
# =====================================================================
def construir_dataset_envolventes(
    df: pd.DataFrame, 
    wl_grid: np.ndarray, 
    prefijo_wl: str = 'nm_', 
    prefijo_r: str = 'R_'
) -> pd.DataFrame:
    """
    Detecta la dimensionalidad del dataset, extrae las matrices y 
    calcula el espectro continuo filtrando solo los descriptores útiles.
    """
    # 1. Detectar transiciones
    cols_wl = [c for c in df.columns if c.startswith(prefijo_wl)]
    cols_r = [c for c in df.columns if c.startswith(prefijo_r)]
    
    num_transiciones = len(cols_wl)
    if num_transiciones == 0 or len(cols_wl) != len(cols_r):
        raise ValueError(f"Error en columnas: Encontradas {len(cols_wl)} WL y {len(cols_r)} R.")
        
    print(f"⚙️ Procesando {df.shape[0]} moléculas con {num_transiciones} transiciones...")
    
    # 2. Extraer matrices y calcular
    wl_matrix = df[cols_wl].values
    r_matrix = df[cols_r].values
    
    espectros_continuos = generar_envolvente_continua(
        wl_transitions=wl_matrix, 
        r_transitions=r_matrix, 
        wl_grid=wl_grid
    )
    
    # 3. FILTRADO POSITIVO: Nos quedamos SOLO con 'Molecula' y descriptores 'Pos_'
    cols_entrada = [c for c in df.columns if c == 'Molecula' or c.startswith('Pos_')]
    df_entradas = df[cols_entrada].reset_index(drop=True)
    
    # 4. Formatear salida
    nombres_y = [f"Y_{np.round(w, 1)}" for w in wl_grid]
    df_salidas = pd.DataFrame(espectros_continuos, columns=nombres_y)
    
    df_final = pd.concat([df_entradas, df_salidas], axis=1)
    
    print(f"✅ Dataset continuo generado: {df_final.shape[0]} filas x {df_final.shape[1]} columnas.")
    return df_final

# =====================================================================
# --- 3. EXPORTACIÓN DINÁMICA ---
# =====================================================================
def exportar_dataset_continuo(
    df_continuo: pd.DataFrame, 
    num_transiciones: int, 
    carpeta_salida: str = "../../data/processed"
) -> str:
    """
    Guarda el DataFrame en un archivo CSV con un nombre dinámico.
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"📁 Carpeta creada automáticamente: {carpeta_salida}")
    
    nombre_archivo = f"Dataset_envolventes_{num_transiciones}T.csv"
    ruta_completa = os.path.join(carpeta_salida, nombre_archivo)
    
    df_continuo.to_csv(ruta_completa, index=False)
    
    print(f"💾 Archivo guardado con éxito: {ruta_completa}")
    return ruta_completa

# =====================================================================
# --- EJECUCIÓN DIRECTA (Opcional) ---
# =====================================================================
if __name__ == "__main__":
    # Malla universal unificada para todo el proyecto
    wl_grid = np.linspace(150, 600, 100)
    
    # Diccionario de configuración de nuestros datasets
    datasets_config = [
        {
            "nombre": "3 Transiciones",
            "ruta": "../../data/raw/DatasetDefinitivo_2310indep.csv",
            "sep": ",",
            "encoding": "ISO-8859-1",
            "prefijo_wl": "nm_R",
            "prefijo_r": "R",
            "n_transiciones": 3
        },
        {
            "nombre": "100 Transiciones",
            "ruta": "../../data/raw/Dataset_ECD_100R.csv",
            "sep": ";",
            "encoding": "utf-8",
            "prefijo_wl": "nm_",
            "prefijo_r": "R_",
            "n_transiciones": 100
        }
    ]
    
    # Pipeline de ejecución por lotes
    print("🚀 Iniciando pipeline de preprocesamiento unificado...\n" + "-"*50)
    
    for config in datasets_config:
        print(f"📁 Analizando dataset: {config['nombre']}")
        
        if os.path.exists(config["ruta"]):
            # Cargar dataset con sus parámetros específicos
            df_crudo = pd.read_csv(config["ruta"], sep=config["sep"], encoding=config["encoding"])
            
            # Generar envolventes
            df_procesado = construir_dataset_envolventes(
                df=df_crudo, 
                wl_grid=wl_grid,
                prefijo_wl=config["prefijo_wl"],
                prefijo_r=config["prefijo_r"]
            )
            
            # Exportar dinámicamente
            exportar_dataset_continuo(
                df_continuo=df_procesado, 
                num_transiciones=config["n_transiciones"],
                carpeta_salida="../../data/processed"
            )
            print("-" * 50)
        else:
            print(f"⚠️ Dataset no encontrado en {config['ruta']}\n" + "-"*50)