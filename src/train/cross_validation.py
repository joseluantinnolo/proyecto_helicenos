"""
Módulo de Validación Cruzada (K-Fold Cross Validation).
Orquesta el entrenamiento de la Red Neuronal dividiendo los datos en K pliegues
para garantizar que las métricas reportadas en el TFG sean estadísticamente significativas.
"""

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

# Importamos nuestras herramientas internas
from src.models.architectures import SpectraPredictorNN
from src.train.engine import train_pinn_model

def run_kfold_cv(
    X_scaled: np.ndarray, 
    Y_scaled: np.ndarray, 
    S_true: np.ndarray, 
    wl_real_t: torch.Tensor,
    criterion_class,
    criterion_kwargs: dict,
    k_folds: int = 5,
    epochs: int = 300,
    batch_size: int = 32,
    device: str = "cpu"
):
    """
    Ejecuta un K-Fold Cross Validation riguroso aislando Train y Val en cada pliegue.
    """
    print(f"\n" + "="*60)
    print(f" 🧪 INICIANDO {k_folds}-FOLD CROSS VALIDATION")
    print("="*60)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    device_obj = torch.device(device)
    
    # Almacén de métricas globales
    fold_metrics = {'val_loss_total': [], 'val_loss_params': [], 'val_loss_spectra': []}
    
    # Aseguramos que X e Y sean tensores de PyTorch
    def to_t(arr): return torch.tensor(arr, dtype=torch.float32)
    X_t, Y_t, S_t = to_t(X_scaled), to_t(Y_scaled), to_t(S_true)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        print(f"\n" + "-"*40)
        print(f" 📁 PLIEGUE (FOLD) {fold} / {k_folds}")
        print("-"*40)
        
        # 1. Creación de Dataloaders para este pliegue específico
        train_ds = TensorDataset(X_t[train_idx], Y_t[train_idx], S_t[train_idx])
        val_ds = TensorDataset(X_t[val_idx], Y_t[val_idx], S_t[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # 2. Instanciación de un modelo NUEVO y VIRGEN para cada fold
        input_dim = X_scaled.shape[1]
        output_dim = Y_scaled.shape[1]
        
        # Arquitectura estándar (se puede parametrizar más adelante con Optuna)
        model = SpectraPredictorNN(input_dim=input_dim, output_dim=output_dim, hidden_layers=[256, 512, 256]).to(device_obj)
        
        # Instanciamos la función de pérdida específica (PINNLoss, EndToEndLoss, etc.)
        criterion = criterion_class(**criterion_kwargs).to(device_obj)
        
        # 3. Lanzamos el motor de entrenamiento que construimos en engine.py
        modelo_entrenado, history = train_pinn_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            wl_real_t=wl_real_t,
            device=device_obj,
            epochs=epochs,
            patience_es=40, # Early stopping moderado para K-Fold
            save_path=f'best_model_fold_{fold}.pt'
        )
        
        # 4. Extracción del mejor resultado de validación de este pliegue
        mejor_val_loss = min(history['val_total'])
        mejor_epoch_idx = history['val_total'].index(mejor_val_loss)
        
        fold_metrics['val_loss_total'].append(mejor_val_loss)
        fold_metrics['val_loss_params'].append(history['val_params'][mejor_epoch_idx])
        fold_metrics['val_loss_spectra'].append(history['val_spectra'][mejor_epoch_idx])
        
        print(f"✅ Fold {fold} Finalizado | Mejor Val Loss: {mejor_val_loss:.4f}")

    # ==========================================
    # --- REPORTE ESTADÍSTICO FINAL ---
    # ==========================================
    print("\n" + "="*60)
    print(" 📊 RESULTADOS FINALES DEL K-FOLD CROSS VALIDATION")
    print("="*60)
    
    mean_total = np.mean(fold_metrics['val_loss_total'])
    std_total = np.std(fold_metrics['val_loss_total'])
    mean_params = np.mean(fold_metrics['val_loss_params'])
    mean_spectra = np.mean(fold_metrics['val_loss_spectra'])
    
    print(f" -> Pérdida Total (Media ± Std): {mean_total:.4f} ± {std_total:.4f}")
    print(f" -> Pérdida Parámetros (Media):  {mean_params:.4f}")
    print(f" -> Pérdida Espectros (Media):   {mean_spectra:.4f}")
    print("="*60)
    
    return fold_metrics