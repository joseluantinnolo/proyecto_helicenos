"""
=============================================================================
ORQUESTADOR MAESTRO - PROYECTO HELICENOS (Dicroísmo Circular)
=============================================================================
Este script es el punto de entrada principal del TFG. 
Soporta TODAS las arquitecturas: Cajas Negras, PINNs de 10 Gaussianas, 
PINNs de 8 Gaussianas y modelos discretos de 3 Transiciones (6 parámetros).
"""

import os
import torch
import numpy as np
import pandas as pd

from src.data.dataloaders import CDDatasetPipeline
from src.models.losses import PINNLoss, EndToEndLoss, DiscretePINNLoss
from src.train.hyperparam_tuning import PINNOptimizer
from src.train.cross_validation import run_kfold_cv
from src.train.train_final import entrenar_modelo_definitivo
from src.evaluate.plots import ScientificPlotter

# =====================================================================
# PANEL DE CONTROL (Configuración de la Ejecución)
# =====================================================================
CONFIG = {
    # MODO DE EJECUCIÓN
    "MODO": "PIPELINE_COMPLETO", 
    
    # ARQUITECTURA OBJETIVO: "PINN_10G", "PINN_8G", "PINN_3T" (Fase A), "CAJA_NEGRA"
    "ARQUITECTURA": "PINN_8G",
    
    # METODO DE EXTRACCIÓN (Solo aplica si no es Caja Negra ni 3T)
    # Opciones: "gmm", "agglomerative", "leastsquares", "pca"
    "METODO": "leastsquares",
    
    "MODELO_NOMBRE": "PINN_LeastSquares_8G_Definitivo",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "N_TRIALS_OPTUNA": 15,
    "MALLA_PUNTOS": 100
}

def cargar_espectros_reales(ruta_csv='data/processed/Dataset_envolventes_100T.csv'):
    if not os.path.exists(ruta_csv): return None
    df = pd.read_csv(ruta_csv)
    cols_y = [c for c in df.columns if c.startswith('Y_')]
    return df[cols_y].values.astype(float)

def configurar_arquitectura():
    """Configura dinámicamente las clases y variables según la arquitectura elegida."""
    arq = CONFIG["ARQUITECTURA"]
    
    if arq == "PINN_10G":
        return PINNLoss, {"num_gaussianas": 10, "a_max": 100.0, "mu_min": 150.0, "mu_max": 650.0, "sigma_max": 60.0}, 10
    
    elif arq == "PINN_8G":
        return PINNLoss, {"num_gaussianas": 8, "a_max": 100.0, "mu_min": 150.0, "mu_max": 650.0, "sigma_max": 60.0}, 8
        
    elif arq == "PINN_3T":
        # Fase A: 3 transiciones (6 parámetros: Amplitud y Posición)
        return DiscretePINNLoss, {"num_transiciones": 3, "lambda_min": 150.0, "lambda_max": 650.0, "r_max": 100.0}, 3
        
    elif arq == "CAJA_NEGRA":
        return EndToEndLoss, {}, None
        
    else:
        raise ValueError(f"Arquitectura {arq} no reconocida.")

def main():
    print("="*70)
    print(f" 🧬 INICIANDO PLATAFORMA DE DICROÍSMO CIRCULAR [{CONFIG['ARQUITECTURA']}] 🧬")
    print("="*70)
    
    wl_grid = np.linspace(150, 600, CONFIG['MALLA_PUNTOS'])
    wl_tensor = torch.tensor(wl_grid, dtype=torch.float32).to(CONFIG['DEVICE'])
    
    # Inyección dinámica de la arquitectura
    CriterionClass, criterion_kwargs, n_gauss = configurar_arquitectura()
    
    X_raw, Y_target, S_true = None, None, None
    best_params = None

    # =====================================================================
    # FASE 1: PREPROCESAMIENTO Y DATALOADERS
    # =====================================================================
    if CONFIG["MODO"] in ["PREPROCESAR", "OPTIMIZAR", "KFOLD", "ENTRENAR_FINAL", "PIPELINE_COMPLETO"]:
        pipeline = CDDatasetPipeline(data_dir="data", batch_size=32)
        S_true = cargar_espectros_reales()
        
        if CONFIG["ARQUITECTURA"] == "CAJA_NEGRA":
            # La Caja Negra intenta predecir el espectro continuo directamente
            X_raw, _ = pipeline.generar_y_guardar_parametros(metodo="pca") # Solo para sacar X_raw
            Y_target = S_true
        else:
            # Para las PINNs, extraemos los parámetros con el número correcto de gaussianas/transiciones
            X_raw, Y_target = pipeline.generar_y_guardar_parametros(
                metodo=CONFIG['METODO'], 
                n_gaussianas=n_gauss
            )
            
        if S_true is None: S_true = np.zeros((len(X_raw), CONFIG['MALLA_PUNTOS']))
        if CONFIG["MODO"] == "PREPROCESAR": return

    # =====================================================================
    # FASE 2: OPTIMIZACIÓN BAYESIANA (OPTUNA)
    # =====================================================================
    if CONFIG["MODO"] in ["OPTIMIZAR", "PIPELINE_COMPLETO"]:
        pipeline = CDDatasetPipeline(data_dir="data", batch_size=32)
        tr_loader, va_loader = pipeline.construir_loaders(X_raw, Y_target, S_true, split_ratio=0.8)
        
        optimizador = PINNOptimizer(
            train_loader=tr_loader, val_loader=va_loader, wl_real_t=wl_tensor, 
            criterion_class=CriterionClass, criterion_kwargs=criterion_kwargs,
            device=CONFIG['DEVICE']
        )
        best_params = optimizador.run_study(n_trials=CONFIG['N_TRIALS_OPTUNA'])
        if CONFIG["MODO"] == "OPTIMIZAR": return
            
    if best_params is None:
        best_params = {"lr": 0.001, "epochs": 300, "n_layers": 3, "layer_0_size": 256, "batch_size": 32}

    # =====================================================================
    # FASE 3: VALIDACIÓN CRUZADA (K-FOLD)
    # =====================================================================
    if CONFIG["MODO"] in ["KFOLD", "PIPELINE_COMPLETO"]:
        pipeline = CDDatasetPipeline(data_dir="data", batch_size=best_params.get('batch_size', 32))
        X_scaled, Y_scaled = pipeline.scaler_X.fit_transform(X_raw), pipeline.scaler_Y.fit_transform(Y_target)
        
        # Le inyectamos la clase de pérdida dinámica
        metricas_kfold = run_kfold_cv(
            X_scaled=X_scaled, Y_scaled=Y_scaled, S_true=S_true, 
            wl_real_t=wl_tensor, criterion_class=CriterionClass, criterion_kwargs=criterion_kwargs,
            k_folds=5, epochs=best_params.get('epochs', 200), batch_size=best_params.get('batch_size', 32), 
            device=CONFIG['DEVICE']
        )
        if CONFIG["MODO"] == "KFOLD": return

    # =====================================================================
    # FASE 4: ENTRENAMIENTO FINAL (PRODUCCIÓN)
    # =====================================================================
    if CONFIG["MODO"] in ["ENTRENAR_FINAL", "PIPELINE_COMPLETO"]:
        modelo, historial = entrenar_modelo_definitivo(
            X_raw=X_raw, Y_target=Y_target, S_true=S_true, wl_grid=wl_grid,
            criterion_class=CriterionClass, criterion_kwargs=criterion_kwargs,
            best_params=best_params, model_name=CONFIG['MODELO_NOMBRE'],
            device=CONFIG['DEVICE']
        )
        
        plotter = ScientificPlotter(output_dir="reports/figures")
        plotter.plot_loss_curves(historial, modelo_nombre=CONFIG['MODELO_NOMBRE'])

if __name__ == "__main__":
    main()