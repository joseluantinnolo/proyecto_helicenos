"""
Módulo de Entrenamiento Final (Producción).
Entrena el modelo definitivo utilizando los hiperparámetros óptimos encontrados
por Optuna y usando la inmensa mayoría de los datos (95% Train).
Soporta de forma 100% dinámica cualquier función de pérdida y arquitectura.
Guarda el modelo compilado listo para inferencia/producción.
"""

import os
import torch
import numpy as np

from src.models.architectures import DynamicPINN
from src.train.engine import train_pinn_model
from src.data.dataloaders import CDDatasetPipeline

def entrenar_modelo_definitivo(
    X_raw: np.ndarray,
    Y_target: np.ndarray,
    S_true: np.ndarray,
    wl_grid: np.ndarray,
    criterion_class,
    criterion_kwargs: dict,
    best_params: dict,
    model_name: str = "Modelo_DC_Definitivo",
    device: str = "cpu"
):
    """
    Entrena y guarda el modelo definitivo en la carpeta `models/`.
    Agnóstico a la arquitectura y función de pérdida gracias a criterion_kwargs.
    
    Args:
        X_raw: Descriptores de entrada (Hammett).
        Y_target: Variable objetivo de salida (pueden ser parámetros o los 100 puntos del espectro).
        S_true: Espectros reales continuos para la pérdida física.
        wl_grid: Malla de longitudes de onda.
        criterion_class: Clase de la pérdida (PINNLoss, EndToEndLoss, DiscretePINNLoss...).
        criterion_kwargs: Diccionario con los parámetros específicos de inicialización de esa pérdida.
        best_params: Hiperparámetros óptimos de la red descubiertos por Optuna.
        model_name: Nombre dinámico para guardar el archivo final sin sobreescribir.
        device: Dispositivo de cómputo ('cpu' o 'cuda').
    """
    print("\n" + "="*60)
    print(f" 🚀 INICIANDO ENTRENAMIENTO FINAL DE PRODUCCIÓN")
    print(f"    -> Nombre del Modelo: {model_name}")
    print("="*60)
    
    # 1. Gestión Dinámica de Rutas y Archivos
    model_dir = "../../models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Limpiamos el nombre para que sea un nombre de archivo seguro
    filename_seguro = model_name.replace(" ", "_").replace("/", "-")
    save_path = os.path.join(model_dir, f"{filename_seguro}.pt")
    
    # 2. Pipeline de Datos Modular (95% Train / 5% Val para Early Stopping)
    pipeline = CDDatasetPipeline(batch_size=best_params.get('batch_size', 32))
    train_loader, val_loader = pipeline.construir_loaders(
        X_raw, Y_target, S_true, split_ratio=0.95
    )
    
    device_obj = torch.device(device)
    input_dim = X_raw.shape[1]
    output_dim = Y_target.shape[1]
    
# 3. Construcción Dinámica de la Topología de la Red Neuronal
    n_layers = best_params.get("n_layers", 3)
        
    model = DynamicPINN(
        input_dim=input_dim, 
        output_dim=output_dim, 
        n_layers=n_layers,
        **{f"layer_{i}_size": best_params.get(f"layer_{i}_size", 256) for i in range(n_layers)}
    ).to(device_obj)
    # 4. INSTANCIACIÓN DINÁMICA UNIVERSAL DE LA FUNCIÓN DE PÉRDIDA
    # Al usar **criterion_kwargs, Python desempaqueta automáticamente los argumentos
    # correctos para PINNLoss, EndToEndLoss o cualquier otra clase sin romper el código.
    criterion = criterion_class(**criterion_kwargs).to(device_obj)
    
    wl_tensor = torch.tensor(wl_grid, dtype=torch.float32).to(device_obj)
    
    # 5. Lanzamiento del Motor Central de Optimización
    lr = best_params.get("lr", 1e-3)
    epochs = best_params.get("epochs", 400)
    weight_decay = best_params.get("weight_decay", 1e-5)
    
    modelo_entrenado, history = train_pinn_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        wl_real_t=wl_tensor,
        device=device_obj,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience_es=50, # Paciencia extendida para el entrenamiento final
        patience_lr=15,
        save_path=save_path
    )
    
    print("\n" + "="*60)
    print(f" 💾 ¡MODELO EN PRODUCCIÓN CONFIGURADO Y SALVADO!")
    print(f"    -> Archivo físico: {save_path}")
    print(f"    -> Capacidad de generalización entrenada con éxito.")
    print("="*60)
    
    return modelo_entrenado, history

# =====================================================================
# EJEMPLOS DE INTEGRACIÓN LÓGICA (Simulación de Inyección Universal)
# =====================================================================
if __name__ == "__main__":
    from src.models.losses import PINNLoss, EndToEndLoss
    
    # Mock de datos sintéticos para validar integridad en terminal
    X_mock = np.random.rand(100, 16)
    wl_mock = np.linspace(150, 600, 100)
    hiperparametros_optima = {"lr": 0.001, "n_layers": 1, "layer_0_size": 128, "epochs": 2}
    
    print("🧪 Test 1: Comprobando dinamismo con modelo PINN (Fase B)...")
    Y_pinn_mock = np.random.rand(100, 30) # 10 Gaussianas
    config_pinn_loss = {
        "num_gaussianas": 10, "a_max": 100.0, "mu_min": 150.0, 
        "mu_max": 650.0, "sigma_max": 60.0, "alpha": 0.6, "beta": 0.4
    }
    entrenar_modelo_definitivo(
        X_mock, Y_pinn_mock, np.random.rand(100, 100), wl_mock,
        criterion_class=PINNLoss, criterion_kwargs=config_pinn_loss,
        best_params=hiperparametros_optima, model_name="PINN_V2.5_GMM"
    )
    
    print("\n🧪 Test 2: Comprobando dinamismo con modelo Caja Negra (End-to-End)...")
    Y_e2e_mock = np.random.rand(100, 100) # Directo a los 100 puntos
    config_e2e_loss = {} # No requiere argumentos en su constructor
    entrenar_modelo_definitivo(
        X_mock, Y_e2e_mock, Y_e2e_mock, wl_mock,
        criterion_class=EndToEndLoss, criterion_kwargs=config_e2e_loss,
        best_params=hiperparametros_optima, model_name="Caja_Negra_Pure_E2E"
    )