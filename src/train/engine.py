"""
Motor de Entrenamiento de Deep Learning.
Contiene el bucle principal de optimización, control de hiperparámetros (Schedulers)
y prevención de sobreajuste (Early Stopping).
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Tuple, Callable
from torch.utils.data import DataLoader

def train_pinn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    wl_real_t: torch.Tensor,
    device: torch.device,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience_es: int = 60,
    patience_lr: int = 15,
    save_path: str = 'best_model.pt'
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Bucle de entrenamiento universal para modelos de Dicroísmo Circular.
    Soporta cualquier arquitectura y función de pérdida instanciada.
    """
    
    # 1. Configuración de Optimizadores
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=patience_lr,
        min_lr=1e-6
    )
    
    # 2. Inicialización de métricas
    history = {
        'total': [], 'params': [], 'spectra': [],
        'val_total': [], 'val_params': [], 'val_spectra': []
    }
    
    paciencia_actual = 0
    mejor_val_loss = float('inf')
    
    print(f'🚀 Iniciando entrenamiento ({epochs} épocas máx) en {device}...')
    print(f'   -> Early Stopping: {patience_es} épocas | LR Scheduler: {patience_lr} épocas')
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        
        # ==========================================
        # --- FASE DE ENTRENAMIENTO ---
        # ==========================================
        model.train()
        ep_total, ep_params, ep_spectra = 0.0, 0.0, 0.0
        
        for bX, bYp, bYs in train_loader:
            bX, bYp, bYs = bX.to(device), bYp.to(device), bYs.to(device)
            
            optimizer.zero_grad()
            pred = model(bX)
            
            # El motor no necesita saber si es la Fase A o B, 
            # solo llama al 'criterion' que le hemos pasado
            loss, lp, ls = criterion(pred, bYp, bYs, wl_real_t)
            
            loss.backward()
            optimizer.step()
            
            ep_total   += loss.item()
            ep_params  += lp.item()
            ep_spectra += ls.item()

        n_tr = len(train_loader)
        history['total'].append(ep_total / n_tr)
        history['params'].append(ep_params / n_tr)
        history['spectra'].append(ep_spectra / n_tr)

        # ==========================================
        # --- FASE DE VALIDACIÓN ---
        # ==========================================
        model.eval()
        vt, vp, vs = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for bX, bYp, bYs in val_loader:
                bX, bYp, bYs = bX.to(device), bYp.to(device), bYs.to(device)
                
                pred = model(bX)
                loss, lp, ls = criterion(pred, bYp, bYs, wl_real_t)
                
                vt += loss.item()
                vp += lp.item()
                vs += ls.item()
                
        n_te = len(val_loader)
        current_val_loss = vt / n_te
        history['val_total'].append(current_val_loss)
        history['val_params'].append(vp / n_te)
        history['val_spectra'].append(vs / n_te)

        # ==========================================
        # --- CONTROL DE ESTADO Y EARLY STOPPING ---
        # ==========================================
        scheduler.step(current_val_loss)
        
        if current_val_loss < mejor_val_loss:
            mejor_val_loss = current_val_loss
            paciencia_actual = 0
            torch.save(model.state_dict(), save_path)
        else:
            paciencia_actual += 1

        # Impresión de progreso limpia (cada 25 épocas)
        if epoch % 25 == 0:
            lr_actual = optimizer.param_groups[0]['lr']
            print(f'Época {epoch:4d} | Tr_Loss: {history["total"][-1]:.4f} | '
                  f'Val_Loss: {history["val_total"][-1]:.4f} | '
                  f'LR: {lr_actual:.1e} | Paciencia: {paciencia_actual}/{patience_es}')

        # Corte de emergencia
        if paciencia_actual >= patience_es:
            print(f'\n[!] 🛑 Early Stopping activado en la época {epoch}.')
            break

    # ==========================================
    # --- CIERRE Y RESTAURACIÓN ---
    # ==========================================
    print('\n✅ ¡Entrenamiento completado!')
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"-> Pesos óptimos restaurados. Mejor Val Loss: {mejor_val_loss:.6f}")
    
    return model, history