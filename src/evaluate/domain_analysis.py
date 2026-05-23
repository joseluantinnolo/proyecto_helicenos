"""
Módulo de Análisis de Dominio de Aplicabilidad y Métricas.
Evalúa el rendimiento de múltiples arquitecturas en función de la complejidad
estérica de las moléculas (Número de Sustituyentes).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import cosine
from scipy.integrate import trapezoid

class DomainEvaluator:
    """
    Analizador universal de métricas físicas y estadísticas.
    """
    def __init__(self, X_data: np.ndarray, wl_nm: np.ndarray):
        """
        X_data: Matriz de entrada (N_muestras, 16). Se usa para contar sustituyentes.
        wl_nm: Malla de longitudes de onda (100,) para integrar el área de error.
        """
        self.X_data = X_data
        self.wl_nm = wl_nm
        # Contamos cuántas posiciones de Hammett son distintas de cero
        self.num_sustituyentes = (self.X_data != 0).sum(axis=1)

    def calcular_metricas_diccionario(self, y_true: np.ndarray, predicciones_dict: dict) -> pd.DataFrame:
        """
        Calcula todas las métricas agrupadas por número de sustituyentes para un número N de modelos.
        predicciones_dict: {'Nombre Modelo': matriz_predicciones_N_x_100}
        """
        print("📊 Iniciando escaneo de dominio de aplicabilidad...")
        resultados = []
        
        for tamano in range(1, 7): # De 1 a 6 sustituyentes
            idx_tamano = np.where(self.num_sustituyentes == tamano)[0]
            
            if len(idx_tamano) < 5: 
                continue # Evitamos ruido estadístico por muestras insuficientes
                
            y_t_subset = y_true[idx_tamano]
            
            fila_resultados = {'Sustituyentes': tamano, 'N_Muestras': len(idx_tamano)}
            
            for nombre_modelo, y_pred in predicciones_dict.items():
                y_p_subset = y_pred[idx_tamano]
                
                # 1. Errores Clásicos
                fila_resultados[f'MAE_{nombre_modelo}'] = mean_absolute_error(y_t_subset, y_p_subset)
                fila_resultados[f'MSE_{nombre_modelo}'] = mean_squared_error(y_t_subset, y_p_subset)
                fila_resultados[f'R2_{nombre_modelo}'] = r2_score(y_t_subset.flatten(), y_p_subset.flatten())
                
                # 2. Métricas Físicas
                cosenos = [1.0 - cosine(y_t_subset[i], y_p_subset[i]) if np.any(y_t_subset[i]) else 0 for i in range(len(y_t_subset))]
                fila_resultados[f'Cos_{nombre_modelo}'] = np.mean(cosenos)
                
                integrales = trapezoid(np.abs(y_p_subset - y_t_subset), x=self.wl_nm, axis=1)
                fila_resultados[f'Int_{nombre_modelo}'] = np.mean(integrales)
                
            resultados.append(fila_resultados)
            
        df_resultados = pd.DataFrame(resultados)
        print("✅ Análisis de sustituyentes completado.")
        return df_resultados

    def generar_codigo_latex(self, df_resultados: pd.DataFrame, modelo_objetivo: str) -> str:
        """
        Genera el código LaTeX inmaculado para la memoria del TFG.
        """
        tabla_latex = pd.DataFrame()
        tabla_latex['Sustituyentes'] = df_resultados['Sustituyentes'].astype(int)
        tabla_latex['Nº Moléculas'] = df_resultados['N_Muestras'].astype(int)
        
        # Formateamos a 3 decimales o sin decimales para la integral
        tabla_latex['MAE'] = df_resultados[f'MAE_{modelo_objetivo}'].apply(lambda x: f"{x:.3f}")
        tabla_latex['R²'] = df_resultados[f'R2_{modelo_objetivo}'].apply(lambda x: f"{x:.3f}")
        tabla_latex['Similitud Coseno'] = df_resultados[f'Cos_{modelo_objetivo}'].apply(lambda x: f"{x:.3f}")
        tabla_latex['Área de Error'] = df_resultados[f'Int_{modelo_objetivo}'].apply(lambda x: f"{x:.0f}")

        latex_code = tabla_latex.to_latex(
            index=False, 
            caption=f"Métricas de rendimiento físico y estadístico del modelo {modelo_objetivo} clasificado por complejidad estérica (Número de sustituyentes activos).",
            label=f"tab:metricas_{modelo_objetivo.lower().replace(' ', '_')}",
            column_format='c|c|cccc',
            escape=False
        )
        return latex_code

    def plot_comparativa_modelos(self, df_resultados: pd.DataFrame, metrica: str, modelos: list):
        """
        Genera un gráfico vectorial comparando los modelos seleccionados para una métrica concreta.
        """
        plt.figure(figsize=(10, 6))
        x = df_resultados['Sustituyentes']
        
        colores = ['royalblue', 'crimson', 'seagreen', 'darkorange', 'purple']
        marcadores = ['o', 's', '^', 'D', 'v']
        
        for i, modelo in enumerate(modelos):
            columna = f"{metrica}_{modelo}"
            if columna in df_resultados.columns:
                plt.plot(x, df_resultados[columna], marker=marcadores[i%len(marcadores)], 
                         color=colores[i%len(colores)], linewidth=2.5, markersize=8, label=modelo)
                
        plt.title(f"Evolución de {metrica} según Complejidad Estérica", fontsize=14, fontweight='bold')
        plt.xlabel("Número de Sustituyentes (Zonas de Hammett activas)", fontsize=12)
        plt.ylabel(metrica, fontsize=12)
        plt.xticks(x)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11)
        
        # Ajuste inteligente de la leyenda si la métrica "más es mejor"
        if metrica in ['R2', 'Cos']:
            plt.legend(loc='lower left')
            
        plt.tight_layout()
        nombre_archivo = f"Comparativa_{metrica}_Sustituyentes.pdf"
        plt.savefig(nombre_archivo, format='pdf', bbox_inches='tight')
        print(f"🎨 Gráfico vectorial guardado como: {nombre_archivo}")
        plt.show()

# =====================================================================
# --- EJEMPLO DE USO DESDE UN NOTEBOOK O SCRIPT MAESTRO ---
# =====================================================================
if __name__ == "__main__":
    # Simulación rápida para comprobar que no hay errores de sintaxis
    print("Iniciando prueba del módulo Evaluador...")
    X_mock = np.random.randint(0, 2, size=(200, 16)) # 200 moléculas, ceros y unos
    wl_mock = np.linspace(150, 600, 100)
    y_true_mock = np.random.rand(200, 100)
    
    # Supongamos que tu motor K-Fold te devolvió estas matrices de predicción:
    preds_dict = {
        "End-to-End": y_true_mock + np.random.normal(0, 0.1, (200, 100)),
        "PINN GMM": y_true_mock + np.random.normal(0, 0.05, (200, 100)),
        "PINN LeastSquares": y_true_mock + np.random.normal(0, 0.02, (200, 100))
    }
    
    evaluador = DomainEvaluator(X_data=X_mock, wl_nm=wl_mock)
    tabla_final = evaluador.calcular_metricas_diccionario(y_true_mock, preds_dict)
    
    # Imprimir LaTeX del mejor modelo
    print("\n" + "="*40 + "\nCÓDIGO LATEX PARA LA MEMORIA\n" + "="*40)
    print(evaluador.generar_codigo_latex(tabla_final, "PINN LeastSquares"))
    
    # Generar gráfico comparativo
    evaluador.plot_comparativa_modelos(tabla_final, metrica="MAE", modelos=list(preds_dict.keys()))