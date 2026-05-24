import optuna
import torch
import numpy as np
from src.models.architectures import SpectraPredictorNN
from src.train.engine import train_pinn_model

class PINNOptimizer:
    def __init__(self, train_loader, val_loader, wl_real_t, criterion_class, criterion_kwargs, device="cpu"):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wl_real_t = wl_real_t
        self.criterion_class = criterion_class
        self.criterion_kwargs = criterion_kwargs # ¡Ahora es universal!
        self.device = torch.device(device)
        
        bX, bYp, bYs = next(iter(train_loader))
        self.input_dim = bX.shape[1]
        self.output_dim = bYp.shape[1]

    def objective(self, trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        # Combinación Convexa
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        beta = 1.0 - alpha
        
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = [trial.suggest_categorical(f"layer_{i}_size", [128, 256, 512, 1024]) for i in range(n_layers)]

        model = SpectraPredictorNN(
            input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=hidden_layers
        ).to(self.device)
        
        # Copiamos los kwargs base (ej. num_gaussianas=8, o num_transiciones=3)
        kwargs_dinamicos = self.criterion_kwargs.copy()
        
        # Si la función de pérdida usa alpha y beta, los actualizamos con los de Optuna
        if "alpha" in kwargs_dinamicos: kwargs_dinamicos["alpha"] = alpha
        if "beta" in kwargs_dinamicos: kwargs_dinamicos["beta"] = beta
            
        criterion = self.criterion_class(**kwargs_dinamicos).to(self.device)

        try:
            _, history = train_pinn_model(
                model=model, train_loader=self.train_loader, val_loader=self.val_loader,
                criterion=criterion, wl_real_t=self.wl_real_t, device=self.device,
                epochs=100, lr=lr, patience_es=20, patience_lr=10, 
                save_path=f"temp_trial_{trial.number}.pt"
            )
            return min(history['val_total'])
        except Exception as e:
            raise optuna.exceptions.TrialPruned()

    def run_study(self, n_trials=50):
        print("\n" + "="*60 + "\n 🎯 INICIANDO OPTIMIZACIÓN BAYESIANA\n" + "="*60)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.objective, n_trials=n_trials)
        print(f"\n🏆 Mejor Loss: {study.best_value:.6f}")
        return study.best_params