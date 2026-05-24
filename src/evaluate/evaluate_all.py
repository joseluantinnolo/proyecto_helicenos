"""
Módulo de Inferencia y Orquestación de Evaluación Multimodelo.
Permite agrupar modelos en "Misiones de Comparación", asegurando que cada 
modelo se evalúe contra su Ground Truth correcto (3T o 100T).
Genera la suite completa de figuras y tablas LaTeX por cada misión.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import cosine
from scipy.integrate import trapezoid

from src.models.architectures import DynamicPINN
from src.data.dataloaders import CDDatasetPipeline
from src.evaluate.plots import ScientificPlotter
from src.evaluate.tables import LaTeXTableGenerator
from src.evaluate.domain_analysis import DomainEvaluator

def ejecutar_evaluacion_comparativa(comparativas_config: list, X_raw: np.ndarray, S_true_3T: np.ndarray, S_true_100T: np.ndarray, wl_grid: np.ndarray, device: str = "cpu"):
    """
    Ejecuta múltiples misiones de evaluación.
    S_true_3T: Espectros continuos generados solo con las 3 transiciones.
    S_true_100T: Espectros continuos generados con las 100 transiciones.
    """
    print("\n" + "="*70)
    print(" 🔎 INICIANDO AUDITORÍA COMPARATIVA (MÚLTIPLES MISIONES)")
    print("="*70)
    
    device_obj = torch.device(device)
    pipeline = CDDatasetPipeline(batch_size=32)
    X_scaled = pipeline.scaler_X.fit_transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device_obj)
    
    plotter = ScientificPlotter(output_dir="reports/figures")
    tabulador = LaTeXTableGenerator(output_dir="reports/tables")
    evaluador_dominio = DomainEvaluator(X_data=X_raw, wl_nm=wl_grid)

    for mision in comparativas_config:
        gt_tipo, nombre_mision, lista_modelos = mision
        
        print(f"\n🚀 EJECUTANDO MISIÓN: {nombre_mision} | Árbitro (Ground Truth): {gt_tipo}")
        
        # Seleccionamos el Árbitro correcto para esta misión
        S_arbitro = S_true_3T if gt_tipo == "3T" else S_true_100T
        
        dict_preds_mision = {}
        dict_resultados_globales = {}
        
        # 1. Bucle de Inferencia
        for mod_name in lista_modelos:
            ruta_pt = f"models/{mod_name}.pt"
            if not os.path.exists(ruta_pt):
                print(f"  ⚠️ No encontrado: {ruta_pt}. Se omite.")
                continue
                
            print(f"  -> Evaluando: {mod_name}...")
            checkpoint = torch.load(ruta_pt, map_location=device_obj)
            
            # Autodescubrimiento de la arquitectura
            pesos_salida = checkpoint[list(checkpoint.keys())[-1]]
            output_dim = pesos_salida.shape[0]
            
            model = DynamicPINN(input_dim=X_raw.shape[1], output_dim=output_dim, hidden_layers=[256, 512, 256]).to(device_obj)
            model.load_state_dict(checkpoint)
            model.eval()
            
            with torch.no_grad():
                y_pred_scaled = model(X_tensor).cpu().numpy()
                
            # Reconstrucción del espectro según el modelo
            if output_dim == 100:
                y_pred_fisico = y_pred_scaled # Caja negra directa
            else:
                pipeline.scaler_Y.fit(np.zeros((10, output_dim))) 
                y_pred_fisico = pipeline.scaler_Y.inverse_transform(y_pred_scaled)
                
                # Renderizar los palitos a espectro continuo
                from src.data.feature_extraction import LeastSquaresExtractor
                espectros_pinn = []
                for idx in range(len(y_pred_fisico)):
                    p = y_pred_fisico[idx]
                    if output_dim == 24: curva = LeastSquaresExtractor.suma_8_gaussianas(wl_grid, *p)
                    elif output_dim == 30: curva = LeastSquaresExtractor.suma_10_gaussianas(wl_grid, *p)
                    elif output_dim == 6: # Fase A (3T): [A, mu], calculamos sigma dinámicamente según la física
                        curva = sum(
                        LeastSquaresExtractor.gaussiana(wl_grid, p[i], p[i+1], ((p[i+1]**2) / 1240.0) * 0.2 + 1e-5) 
                        for i in range(0, 6, 2)
                        )
                    else: curva = np.zeros(100)
                    espectros_pinn.append(curva)
                y_pred_fisico = np.array(espectros_pinn)
            
            dict_preds_mision[mod_name] = y_pred_fisico
            
            # --- CÁLCULO DE TODAS LAS MÉTRICAS GLOBALES ---
            mae_glob = mean_absolute_error(S_arbitro, y_pred_fisico)
            mse_glob = mean_squared_error(S_arbitro, y_pred_fisico)
            r2_glob = r2_score(S_arbitro.flatten(), y_pred_fisico.flatten())
            cosenos = [1.0 - cosine(S_arbitro[i], y_pred_fisico[i]) if np.any(S_arbitro[i]) else 0 for i in range(len(S_arbitro))]
            integrales = trapezoid(np.abs(y_pred_fisico - S_arbitro), x=wl_grid, axis=1)
            
            dict_resultados_globales[mod_name] = {
                "MAE": mae_glob,
                "MSE": mse_glob,
                "R$^2$": r2_glob,
                "Sim. Coseno": np.mean(cosenos),
                "Área Error (Int)": np.mean(integrales)
            }

        if not dict_preds_mision: continue

        # =====================================================================
        # 2. GENERACIÓN AUTOMÁTICA DE TABLAS Y GRÁFICOS (Conservando todo)
        # =====================================================================
        print(f"  🎨 Exportando Tablas y Gráficos de la misión: {nombre_mision}...")
        
        # T3: Tabla comparativa global para LaTeX
        tabulador.generar_tabla_comparativa(dict_resultados_globales, suffix_nombre=nombre_mision)
        
        # T4: Tabla de Dominio por Sustituyentes (LaTeX)
        df_dominio = evaluador_dominio.calcular_metricas_diccionario(S_arbitro, dict_preds_mision)
        for mod_name in dict_preds_mision.keys():
            latex_dominio = evaluador_dominio.generar_codigo_latex(df_dominio, mod_name)
            with open(f"reports/tables/T4_Sustituyentes_{mod_name}_{nombre_mision}.tex", "w") as f:
                f.write(latex_dominio)
        
        # G2: Superposición Multimodelo (Molécula 0 como ejemplo)
        plotter.plot_comparativa_modelos_espectro(wl_grid, S_arbitro[0], dict_preds_mision, mol_idx=0, suffix_nombre=nombre_mision)
        
        # G3: Grid de 6x4 por Sustituyentes
        num_sustituyentes = (X_raw != 0).sum(axis=1)
        plotter.plot_grid_inspeccion_sustituyentes(wl_grid, S_arbitro, dict_preds_mision, num_sustituyentes, np.arange(len(X_raw)), modelo_nombre=nombre_mision)

    print("\n✅ ¡TODAS LAS MISIONES DE COMPARACIÓN FINALIZADAS! Revisa reports/tables y reports/figures.")