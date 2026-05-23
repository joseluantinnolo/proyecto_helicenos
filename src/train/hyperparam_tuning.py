"""
Módulo de Optimización de Hiperparámetros (Bayesian Tuning).
Utiliza Optuna para encontrar el balance perfecto entre la física (beta) 
y los parámetros (alpha) en la función de pérdida PINN, además de optimizar la red.
"""

import optuna
import torch
import numpy as np
from typing import Callable

# Importamos nuestra arquitectura y el motor
from src.models.architectures import SpectraPredictorNN
from src.train.engine import train_pinn_model

class PINNOptimizer:
    """
    Orquestador de Optuna para modelos informados por la física.
    """
    def __init__(self, train_loader, val_loader, wl_real_t, criterion_class, device="cpu"):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wl_real_t = wl_real_t
        self.criterion_class = criterion_class
        self.device = torch.device(device)
        
        # Obtenemos las dimensiones automáticamente desde un batch
        bX, bYp, bYs = next(iter(train_loader))
        self.input_dim = bX.shape[1]
        self.output_dim = bYp.shape[1]

    def objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo que Optuna intentará minimizar.
        """
        # 1. ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        # EL BALANCE FÍSICO (Combinación Convexa: alpha + beta = 1)
        # Optuna buscará un valor fino (ej. 0.83), no solo saltos de 0.1
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        beta = 1.0 - alpha
        
        # Profundidad de la red
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = []
        for i in range(n_layers):
            size = trial.suggest_categorical(f"layer_{i}_size", [128, 256, 512, 1024])
            hidden_layers.append(size)

        # 2. INSTANCIACIÓN DINÁMICA
        model = SpectraPredictorNN(
            input_dim=self.input_dim, 
            output_dim=self.output_dim, 
            hidden_layers=hidden_layers
        ).to(self.device)
        
        # Le pasamos los pesos sugeridos a nuestra función de pérdida
        criterion = self.criterion_class(
            num_gaussianas=(self.output_dim // 3), # Asumiendo 3 parámetros por gaussiana
            a_max=100.0, mu_min=150.0, mu_max=650.0, sigma_max=60.0, # Ajustar si es necesario
            alpha=alpha, 
            beta=beta
        ).to(self.device)

        # 3. ENTRENAMIENTO RÁPIDO (Solo 100 épocas para ver el potencial)
        try:
            _, history = train_pinn_model(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                criterion=criterion,
                wl_real_t=self.wl_real_t,
                device=self.device,
                epochs=100,  # Épocas reducidas para la fase de búsqueda
                lr=lr,
                patience_es=20,
                patience_lr=10,
                save_path=f"temp_trial_{trial.number}.pt"
            )
            
            # El valor a minimizar es la mejor pérdida de validación obtenida
            mejor_val_loss = min(history['val_total'])
            return mejor_val_loss
            
        except Exception as e:
            # Si una combinación inestable hace que el gradiente explote (NaN),
            # le decimos a Optuna que aborte este intento y lo marque como fallido.
            raise optuna.exceptions.TrialPruned()

    def run_study(self, n_trials=50):
        """Lanza el estudio completo de optimización."""
        print("\n" + "="*60)
        print(" 🎯 INICIANDO OPTIMIZACIÓN BAYESIANA CON OPTUNA")
        print("="*60)
        
        # El TPESampler es el motor probabilístico por debajo de Optuna
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.objective, n_trials=n_trials)
        
        print("\n" + "="*60)
        print(" 🏆 MEJOR COMBINACIÓN DE HIPERPARÁMETROS ENCONTRADA")
        print("="*60)
        print(f"Mejor Loss: {study.best_value:.6f}")
        for key, value in study.best_params.items():
            print(f" -> {key}: {value}")
            
        return study.best_params