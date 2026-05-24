"""
Módulo de Generación de Tablas LaTeX (Reportes).
Convierte los resultados de evaluación y los hiperparámetros en código LaTeX
puro, ajustado automáticamente al ancho de la página de la memoria del TFG.
"""

import os
import numpy as np
import pandas as pd

class LaTeXTableGenerator:
    """Clase maestra para generar tablas científicas listas para compilar."""
    
    def __init__(self, output_dir="../../reports/tables"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _exportar_con_ajuste_ancho(self, df: pd.DataFrame, filename: str, caption: str, label: str):
        """
        Exporta el DataFrame a LaTeX y le inyecta el \resizebox para que 
        no se salga de los márgenes de la página.
        """
        latex_str = df.to_latex(index=False, escape=False, column_format='c' * len(df.columns))
        
        # Inyección del parche de ancho de página
        latex_str = latex_str.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}")
        latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}%\n}")
        
        entorno_completo = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"{latex_str}"
            "\\end{table}\n"
        )
        
        ruta = os.path.join(self.output_dir, filename)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write(entorno_completo)
            
        print(f"📄 Tabla LaTeX guardada: {ruta}")

    def generar_tabla_hiperparametros(self, best_params: dict, modelo_nombre: str):
        """G1: Tabla de parámetros óptimos de Optuna."""
        if not best_params: return
            
        filas = []
        for k, v in best_params.items():
            nombre_limpio = str(k).replace("_", " ").title()
            if isinstance(v, float):
                valor_str = f"{v:.1e}" if v < 0.01 else f"{v:.4f}"
            else:
                valor_str = str(v)
            filas.append({"Hiperparámetro": nombre_limpio, "Valor Óptimo": valor_str})
            
        df = pd.DataFrame(filas)
        self._exportar_con_ajuste_ancho(
            df, 
            filename=f"T1_Hiperparams_{modelo_nombre}.tex", 
            caption=f"Hiperparámetros óptimos (Optuna) para {modelo_nombre.replace('_', ' ')}.", 
            label=f"tab:params_{modelo_nombre.lower()}"
        )

    def generar_tabla_kfold(self, metricas_kfold: dict, modelo_nombre: str):
        """G2: Resultados de los 5 Folds con Media +- Desviación."""
        if not metricas_kfold: return
            
        filas = []
        for metrica, valores in metricas_kfold.items():
            if len(valores) == 0: continue
            media, std = np.mean(valores), np.std(valores)
            nombre_limpio = metrica.replace("_", " ").title()
            filas.append({
                "Métrica": nombre_limpio,
                "Media ($\\mu$)": f"{media:.4f}",
                "Desv. Típica ($\\sigma$)": f"{std:.4f}",
                "Reporte Final": f"${media:.4f} \\pm {std:.4f}$"
            })
            
        df = pd.DataFrame(filas)
        self._exportar_con_ajuste_ancho(
            df, 
            filename=f"T2_KFold_{modelo_nombre}.tex", 
            caption=f"Validación Cruzada (5-Fold) para {modelo_nombre.replace('_', ' ')}.", 
            label=f"tab:kfold_{modelo_nombre.lower()}"
        )

    def generar_tabla_comparativa(self, dict_resultados_globales: dict, nombre_mision: str):
        """G3: Enfrentamiento directo de varios modelos."""
        df = pd.DataFrame.from_dict(dict_resultados_globales, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Arquitectura'}, inplace=True)
        
        for col in df.columns:
            if df[col].dtype == float:
                df[col] = df[col].apply(lambda x: f"{x:.3f}")
                
        self._exportar_con_ajuste_ancho(
            df, 
            filename=f"T3_Comparativa_{nombre_mision}.tex", 
            caption=f"Comparativa de rendimiento: {nombre_mision.replace('_', ' ')}.", 
            label=f"tab:comp_{nombre_mision.lower()}"
        )