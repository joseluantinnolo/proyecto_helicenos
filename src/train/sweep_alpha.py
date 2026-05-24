"""
Módulo de Análisis Aislado: Barrido de Hiperparámetros Físicos (Alpha/Beta).
Ejecuta el entrenamiento secuencial modificando el peso de la función de pérdida
para justificar la elección del "Sweet Spot" (Alpha = 0.2) en la memoria del TFG.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import cosine
from scipy.integrate import trapezoid

from src.models.architectures import DynamicPINN
from src.train.engine import train_pinn_model

def evaluate_physical_metrics(S_true, S_pred, wl_nm):
    """Calcula MAE, MSE, RMSE, R2, Coseno e Integral del Error del espectro completo."""
    mae = mean_absolute_error(S_true, S_pred)
    mse = mean_squared_error(S_true, S_pred)
    r2 = r2_score(S_true.flatten(), S_pred.flatten()) 
    
    cosenos = [1.0 - cosine(S_true[i], S_pred[i]) if np.any(S_true[i]) else 0 for i in range(len(S_true))]
    areas_discrepancia = trapezoid(np.abs(S_pred - S_true), x=wl_nm, axis=1)
    
    return {
        'MAE': mae, 'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2,
        'Coseno_Medio': np.mean(cosenos),
        'Integral_Error_Media': np.mean(areas_discrepancia)
    }

def ejecutar_barrido_alpha(train_loader, val_loader, wl_grid, input_dim, output_dim, 
                           criterion_class, criterion_kwargs_base, device_str="cpu"):
    
    print("\n" + "="*70)
    print(" 🚀 INICIANDO BARRIDO AISLADO DE ALPHA / BETA")
    print("="*70)
    
    device = torch.device(device_str)
    wl_tensor = torch.tensor(wl_grid, dtype=torch.float32).to(device)
    
    resultados_barrido = []
    alphas = [round(a, 1) for a in np.arange(1.0, -0.1, -0.1)]
    
    # Extraemos matrices completas de validación para evaluar al final
    bX_val, _, bYs_val = val_loader.dataset.tensors
    S_true_val = bYs_val.numpy()

    for alpha in alphas:
        beta = round(1.0 - alpha, 1)
        print(f"\n⚙️ ENTRENANDO: Alpha (Params)={alpha} | Beta (Espectro)={beta}")
        
        # 1. Configurar Pérdida y Modelo (Igual que tu cuaderno)
        kwargs = criterion_kwargs_base.copy()
        kwargs['alpha'] = alpha
        kwargs['beta'] = beta
        criterion = criterion_class(**kwargs).to(device)
        
        torch.manual_seed(42)
        model = DynamicPINN(
            input_dim=input_dim, output_dim=output_dim, 
            n_layers=3, layer_0_size=128, layer_1_size=64, layer_2_size=32
        ).to(device)
        
        # 2. Entrenar (Limitado a 150 épocas para el barrido para ir rápido, ajustable)
        start_time = time.time()
        modelo_entrenado, _ = train_pinn_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, wl_real_t=wl_tensor, device=device,
            epochs=150, lr=1e-3, patience_es=30, patience_lr=10, 
            save_path="temp_sweep_model.pt"
        )
        tiempo_total = time.time() - start_time
        
        # 3. Evaluar Espectros
        modelo_entrenado.eval()
        with torch.no_grad():
            pred_norm = modelo_entrenado(bX_val.to(device))
            # Usamos la función interna de la pérdida para reconstruir el espectro físico
            S_pred_norm = criterion.build_spectrum(pred_norm, wl_tensor)
            
            # Desescalamos a intensidad real (multiplicando por a_max o r_max)
            escala = kwargs.get('a_max', kwargs.get('r_max', 100.0))
            S_pred_real = (S_pred_norm * escala).cpu().numpy()
            
        metricas = evaluate_physical_metrics(S_true_val, S_pred_real, wl_grid)
        
        resultados_barrido.append({
            'Alpha_Params': alpha, 'Beta_Espectro': beta, 
            'Tiempo_Minutos': round(tiempo_total / 60, 2), **metricas
        })
        print(f"✅ Iteración completada. Coseno obtenido: {metricas['Coseno_Medio']:.4f}")

    # =========================================================
    # GUARDADO DE DATOS Y GRÁFICAS (Tus gráficas exactas)
    # =========================================================
    os.makedirs("reports/analisis_extra", exist_ok=True)
    df = pd.DataFrame(resultados_barrido)
    df.to_csv('reports/analisis_extra/barrido_alpha_beta.csv', index=False)
    
    # Imprimir LaTeX
    print("\n📄 CÓDIGO LATEX DE LA TABLA (Listo para Overleaf)")
    print(df.to_latex(index=False, float_format="%.4f"))
    
    # Dibujar la figura 2x2
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evolución de las Métricas vs. Peso de la Función de Pérdida', fontsize=18, fontweight='bold', y=1.02)

    metricas_plot = [
        {'columna': 'R2', 'ax': axes[0, 0], 'titulo': 'Coeficiente de Determinación ($R^2$)', 'color': '#1f77b4', 'mejor': 'arriba'},
        {'columna': 'Coseno_Medio', 'ax': axes[0, 1], 'titulo': 'Similitud del Coseno Media', 'color': '#2ca02c', 'mejor': 'arriba'},
        {'columna': 'MAE', 'ax': axes[1, 0], 'titulo': 'Error Absoluto Medio (MAE)', 'color': '#d62728', 'mejor': 'abajo'},
        {'columna': 'Integral_Error_Media', 'ax': axes[1, 1], 'titulo': 'Dif. Absoluta Integrada (IAD)', 'color': '#9467bd', 'mejor': 'abajo'}
    ]

    for m in metricas_plot:
        ax = m['ax']
        ax.plot(df['Alpha_Params'], df[m['columna']], marker='o', linewidth=2, markersize=8, color=m['color'])
        ax.set_xlim(1.05, -0.05)
        ax.set_title(m['titulo'], fontweight='bold')
        ax.set_xlabel('Peso de los Parámetros ($\\alpha$) $\\rightarrow$ Mayor peso al espectro')
        ax.set_ylabel('Valor de la métrica')
        ax.axvspan(0.25, 0.05, color='gold', alpha=0.2, label='Zona Óptima')
        
        if m['mejor'] == 'arriba':
            mejor_val, mejor_alpha = df[m['columna']].max(), df.loc[df[m['columna']].idxmax(), 'Alpha_Params']
        else:
            mejor_val, mejor_alpha = df[m['columna']].min(), df.loc[df[m['columna']].idxmin(), 'Alpha_Params']
            
        ax.scatter(mejor_alpha, mejor_val, color='gold', edgecolor='black', s=200, marker='*', zorder=5, label='Mejor Valor')
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig('reports/analisis_extra/grafica_barrido_metricas.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    print("🎉 ¡BARRIDO FINALIZADO! Gráficas en reports/analisis_extra/grafica_barrido_metricas.pdf")