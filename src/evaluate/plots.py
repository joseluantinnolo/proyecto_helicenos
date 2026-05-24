"""
Módulo de Visualización Científica (El Atlas Visual).
Genera gráficos vectoriales con calidad de publicación (PDF, 300 dpi)
con nombres dinámicos para evitar sobreescritura entre modelos.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random

# Configuración global para estilo revista científica
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})

class ScientificPlotter:
    """Generador centralizado de figuras para la memoria del TFG."""
    
    def __init__(self, output_dir="../../reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_path(self, prefix: str, modelo_nombre: str, suffix: str = "") -> str:
        """Genera un nombre de archivo dinámico y seguro."""
        nombre_limpio = modelo_nombre.replace(" ", "_").replace("/", "-")
        if suffix:
            filename = f"{prefix}_{nombre_limpio}_{suffix}.pdf"
        else:
            filename = f"{prefix}_{nombre_limpio}.pdf"
        return os.path.join(self.output_dir, filename)

    # =====================================================================
    # 1. CURVAS DE APRENDIZAJE (Inteligentes)
    # =====================================================================
    def plot_loss_curves(self, history: dict, modelo_nombre: str):
        """
        Dibuja 1 gráfica para Caja Negra o 3 subgráficas para PINNs.
        """
        epochs = range(1, len(history['total']) + 1)
        es_pinn = np.sum(history['params']) > 0  # Si hay error paramétrico, es PINN

        if es_pinn:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            metricas = [('total', 'val_total', 'Pérdida Total (Híbrida)'),
                        ('params', 'val_params', 'Error Paramétrico (MSE)'),
                        ('spectra', 'val_spectra', 'Error Espectral (Física)')]
            
            for i, (tr_key, val_key, title) in enumerate(metricas):
                axes[i].plot(epochs, history[tr_key], label='Train', color='royalblue')
                axes[i].plot(epochs, history[val_key], label='Validation', color='crimson')
                best_epoch = np.argmin(history[val_key]) + 1
                axes[i].axvline(x=best_epoch, color='black', linestyle='--', alpha=0.5)
                axes[i].set_title(title, fontweight='bold')
                axes[i].set_xlabel('Épocas')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            plt.suptitle(f'Dinámica de Entrenamiento PINN: {modelo_nombre}', fontsize=16, y=1.05)
        else:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, history['total'], label='Train Loss', color='black')
            plt.plot(epochs, history['val_total'], label='Validation Loss', color='red')
            best_epoch = np.argmin(history['val_total']) + 1
            plt.axvline(x=best_epoch, color='black', linestyle='--', alpha=0.5, label=f'Best ({best_epoch})')
            plt.title(f'Entrenamiento Caja Negra: {modelo_nombre}', fontweight='bold')
            plt.xlabel('Épocas')
            plt.ylabel('MSE')
            plt.grid(True, alpha=0.3)
            plt.legend()

        ruta = self._get_path("G1_Loss", modelo_nombre)
        plt.savefig(ruta)
        plt.close()
        print(f"✅ Loss Curve guardada: {ruta}")

    # =====================================================================
    # 2. COMPARATIVA MULTIMODELO EN UNA MOLÉCULA
    # =====================================================================
    def plot_comparativa_modelos_espectro(self, wl_nm: np.ndarray, y_real: np.ndarray, 
                                          dict_preds: dict, mol_idx: int = 0):
        """
        Superpone las predicciones de varios modelos (End-to-End, PINN, etc.) 
        contra la curva real (Ground Truth).
        """
        plt.figure(figsize=(10, 6))
        plt.plot(wl_nm, y_real, color='black', linewidth=3, label='Real (Ground Truth)')
        
        colores = ['crimson', 'royalblue', 'seagreen', 'darkorange']
        estilos = ['--', '-.', ':', '--']
        
        for i, (nombre_mod, y_pred) in enumerate(dict_preds.items()):
            c = colores[i % len(colores)]
            s = estilos[i % len(estilos)]
            plt.plot(wl_nm, y_pred, color=c, linestyle=s, linewidth=2, label=nombre_mod)

        plt.axhline(0, color='black', linewidth=1)
        plt.title(f'Comparativa de Modelos (Molécula ID: {mol_idx})', fontweight='bold')
        plt.xlabel('Longitud de Onda (nm)')
        plt.ylabel('Intensidad CD (Δε)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        ruta = self._get_path("G2_Comparativa_Multimodelo", f"Mol_{mol_idx}")
        plt.savefig(ruta)
        plt.close()
        print(f"✅ Comparativa multimodelo guardada: {ruta}")

    # =====================================================================
    # 3. GRID 6x4 (INSPECCIÓN VISUAL POR SUSTITUYENTES)
    # =====================================================================
    def plot_grid_inspeccion_sustituyentes(self, wl_nm: np.ndarray, y_real: np.ndarray, 
                                           dict_preds: dict, num_sustituyentes: np.ndarray, 
                                           test_idx: np.ndarray, modelo_nombre: str = "Multimodelo"):
        """
        Matriz de 6 filas (1 a 6 sustituyentes) x 4 columnas (ejemplos aleatorios).
        """
        fig, axes = plt.subplots(6, 4, figsize=(20, 24))
        random.seed(42) # Semilla fija para reproducibilidad en la memoria
        
        colores = ['royalblue', 'crimson', 'seagreen']
        estilos = ['--', '-.', ':']

        for tamano in range(1, 7):
            idx_tamano = test_idx[num_sustituyentes[test_idx] == tamano]
            n_ejemplos = min(4, len(idx_tamano))
            ejemplos = random.sample(list(idx_tamano), n_ejemplos)
            
            for col, idx_mol in enumerate(ejemplos):
                ax = axes[tamano-1, col]
                ax.plot(wl_nm, y_real[idx_mol], 'k-', linewidth=2.5, label='Exacto')
                
                for i, (nom_mod, y_preds_mod) in enumerate(dict_preds.items()):
                    ax.plot(wl_nm, y_preds_mod[idx_mol], color=colores[i%len(colores)], 
                            linestyle=estilos[i%len(estilos)], linewidth=1.5, label=nom_mod)
                
                ax.set_title(f"Sust: {tamano} | Mol: {idx_mol}", fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                if col == 0: ax.set_ylabel('Intensidad CD')
                if tamano == 6: ax.set_xlabel('nm')
                if tamano == 1 and col == 0: ax.legend()

        # Limpiar ejes vacíos
        for tamano in range(1, 7):
            idx_tamano = test_idx[num_sustituyentes[test_idx] == tamano]
            for col in range(min(4, len(idx_tamano)), 4):
                fig.delaxes(axes[tamano-1, col])

        plt.suptitle(f"Inspección Visual (Test Fold 1) vs Complejidad Estérica", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        ruta = self._get_path("G3_Grid_6x4", modelo_nombre)
        plt.savefig(ruta)
        plt.close()
        print(f"✅ Grid 6x4 guardado: {ruta}")

    # =====================================================================
    # 4. DISCRETO VS CONTINUO (Palitos vs Envolvente Gaussiana)
    # =====================================================================
    def plot_discreto_vs_continuo(self, wl_grid: np.ndarray, wl_sticks: np.ndarray, 
                                  r_sticks: np.ndarray, y_envolvente: np.ndarray, 
                                  mol_idx: int = 0, modelo_nombre: str = "GroundTruth"):
        """
        Compara los estados excitados cuánticos (palitos) con la envolvente continua generada.
        """
        plt.figure(figsize=(10, 5))
        
        # Filtramos NaNs de los palitos
        valid_idx = ~np.isnan(wl_sticks) & ~np.isnan(r_sticks)
        w_val = wl_sticks[valid_idx]
        r_val = r_sticks[valid_idx]
        
        # Dibujamos las transiciones discretas (Sticks)
        plt.stem(w_val, r_val, linefmt='grey', markerfmt='ko', basefmt=" ", 
                 label=f'Transiciones Discretas (N={len(w_val)})')
        
        # Dibujamos la envolvente continua
        plt.plot(wl_grid, y_envolvente, color='crimson', linewidth=2.5, label='Envolvente Continua (Suma Gaussiana)')
        
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f'Estados Excitados Discretos vs Espectro ECD Continuo (Molécula {mol_idx})', fontweight='bold')
        plt.xlabel('Longitud de Onda (nm)')
        plt.ylabel('Fuerza Rotatoria / Intensidad CD')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        ruta = self._get_path("G4_Discreto_Vs_Continuo", modelo_nombre, f"Mol_{mol_idx}")
        plt.savefig(ruta)
        plt.close()
        print(f"✅ Gráfico Discreto vs Continuo guardado: {ruta}")

    # =====================================================================
    # 5. COMPARACIÓN: 3 TRANSICIONES VS 100 TRANSICIONES
    # =====================================================================
    def plot_3_vs_100_transiciones(self, wl_nm: np.ndarray, y_3: np.ndarray, y_100: np.ndarray, mol_idx: int = 0):
        """Muestra el salto de calidad entre la base de datos de 3T y la de 100T."""
        plt.figure(figsize=(10, 5))
        plt.plot(wl_nm, y_3, color='royalblue', linestyle='--', linewidth=2, label='Espectro Simplificado (3 Transiciones)')
        plt.plot(wl_nm, y_100, color='black', linewidth=2.5, label='Espectro Realista (100 Transiciones)')
        
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f'Impacto del Nivel de Teoría: 3 vs 100 Transiciones (Molécula {mol_idx})', fontweight='bold')
        plt.xlabel('Longitud de Onda (nm)')
        plt.ylabel('Intensidad CD')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        ruta = self._get_path("G5_Comparativa_3_vs_100", f"Mol_{mol_idx}")
        plt.savefig(ruta)
        plt.close()
        print(f"✅ Comparativa 3vs100 guardada: {ruta}")

    # =====================================================================
    # 6. HELICENO BASE VS ALEATORIOS
    # =====================================================================
    def plot_heliceno_vs_aleatorios(self, wl_nm: np.ndarray, y_base: np.ndarray, 
                                    matriz_y_aleatorios: np.ndarray, num_aleatorios: int = 3):
        """
        Compara la estructura del Heliceno sin sustituyentes (esqueleto base) 
        con moléculas altamente sustituidas para ver la distorsión del espectro.
        """
        plt.figure(figsize=(10, 5))
        
        # Dibujamos las aleatorias primero (fondo)
        colores_rand = ['seagreen', 'darkorange', 'purple', 'gray']
        indices_rand = random.sample(range(matriz_y_aleatorios.shape[0]), num_aleatorios)
        
        for i, idx in enumerate(indices_rand):
            plt.plot(wl_nm, matriz_y_aleatorios[idx], color=colores_rand[i%len(colores_rand)], 
                     alpha=0.6, linestyle='-.', label=f'Molécula Aleatoria (ID {idx})')
            
        # Dibujamos la base (fuerte y por encima)
        plt.plot(wl_nm, y_base, color='black', linewidth=3, label='Heliceno Base (Sin sustituyentes)')
        
        plt.axhline(0, color='black', linewidth=1)
        plt.title('Distorsión del Espectro ECD por efecto de Sustituyentes', fontweight='bold')
        plt.xlabel('Longitud de Onda (nm)')
        plt.ylabel('Intensidad CD')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        ruta = self._get_path("G6_Base_vs_Aleatorios", "Analisis_Topologico")
        plt.savefig(ruta)
        plt.close()
        print(f"✅ Gráfico Base vs Aleatorios guardado: {ruta}")

    # =====================================================================
    # 7. DIAGRAMAS DE CODO (Para K-Means, GMM o nº de Gaussianas)
    # =====================================================================
    def plot_diagrama_codo(self, k_values: list, metric_values: list, 
                           ylabel: str = "Inercia / Error (MSE)", modelo_nombre: str = "Optimizacion_K"):
        """Dibuja el clásico diagrama de codo para justificar hiperparámetros."""
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, metric_values, marker='o', color='navy', linewidth=2, markersize=8)
        
        plt.title(f'Diagrama de Codo: Optimización de Hiperparámetros ({modelo_nombre})', fontweight='bold')
        plt.xlabel('Número de Componentes / Gaussianas (K)')
        plt.ylabel(ylabel)
        
        # Forzar que el eje X muestre números enteros
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True, alpha=0.3, linestyle='--')
        
        ruta = self._get_path("G7_Diagrama_Codo", modelo_nombre)
        plt.savefig(ruta)
        plt.close()
        print(f"✅ Diagrama de Codo guardado: {ruta}")

    # =====================================================================
    # 8. PARKING DE ZOMBIS Y PARIDAD (Ya los teníamos, actualizados con nombre)
    # =====================================================================
    def plot_parking_zombis(self, params_pred: np.ndarray, modelo_nombre: str):
        plt.figure(figsize=(10, 6))
        A_todas = params_pred[:, 0::3].flatten()
        mu_todas = params_pred[:, 1::3].flatten()
        mask_zombis = np.abs(A_todas) < 1.0
        
        plt.scatter(mu_todas[mask_zombis], A_todas[mask_zombis], color='gray', alpha=0.2, s=20, label='Zombis (Apagadas)')
        scatter = plt.scatter(mu_todas[~mask_zombis], A_todas[~mask_zombis], c=A_todas[~mask_zombis], cmap='coolwarm', alpha=0.7, s=30, label='Activas')
        
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.title(f'Mapa del Espacio Latente (Efectos Cotton) - {modelo_nombre}', fontweight='bold')
        plt.xlabel('Longitud de Onda - $\mu$ (nm)')
        plt.ylabel('Amplitud de la Gaussiana - $A$')
        plt.colorbar(scatter, label="Intensidad")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        ruta = self._get_path("G8_Parking_Zombis", modelo_nombre)
        plt.savefig(ruta)
        plt.close()

    def plot_paridad_picos(self, picos_reales: np.ndarray, picos_predichos: np.ndarray, modelo_nombre: str):
        plt.figure(figsize=(7, 7))
        min_val, max_val = min(np.min(picos_reales), np.min(picos_predichos)), max(np.max(picos_reales), np.max(picos_predichos))
        limites = [min_val - (max_val - min_val)*0.1, max_val + (max_val - min_val)*0.1]
        
        plt.plot(limites, limites, color='black', linestyle='--', label='Precisión Perfecta (y=x)')
        plt.scatter(picos_reales, picos_predichos, color='seagreen', alpha=0.6, edgecolor='white')
        
        plt.title(f'Gráfico de Paridad de Picos - {modelo_nombre}', fontweight='bold')
        plt.xlabel('Intensidad CD Real')
        plt.ylabel('Intensidad CD Predicha')
        plt.xlim(limites); plt.ylim(limites)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        ruta = self._get_path("G9_Paridad", modelo_nombre)
        plt.savefig(ruta)
        plt.close()