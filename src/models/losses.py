"""
Módulo de Funciones de Pérdida y Física (PINN).
Contiene la reconstrucción analítica del espectro y la función de pérdida híbrida.
"""

import torch
import torch.nn as nn
# =====================================================================
# --- PINN FASE A: TRANSICIONES DISCRETAS (3 Picos) ---
# =====================================================================
class DiscretePINNLoss(nn.Module):
    """
    Función de pérdida física para el modelo de 3 transiciones (6 parámetros).
    A diferencia del modelo de 100R, aquí la anchura (sigma) no es predicha por la red,
    sino derivada analíticamente de la longitud de onda (lambda) dentro del grafo.
    """
    def __init__(self, 
                 num_transiciones: int,
                 lambda_min: float, 
                 lambda_max: float, 
                 r_max: float, 
                 sigma_ev: float = 0.2,
                 alpha: float = 1.0, 
                 beta: float = 1.0):
        super(DiscretePINNLoss, self).__init__()
        self.num_transiciones = num_transiciones
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.r_max = r_max
        self.sigma_ev = sigma_ev
        self.alpha = alpha
        self.beta = beta
        
        # Constantes Físicas
        self.hc_ev_nm = 1240.0
        self.epsilon = 1e-5
        
        self.mse = nn.MSELoss()

    def denorm_lambda(self, v: torch.Tensor) -> torch.Tensor:
        """Desnormaliza asumiendo que v viene en rango [-1, 1] o [0, 1] (ajustar según tu pipeline)"""
        # Asumiendo entrada [0, 1] para la posición:
        return v * (self.lambda_max - self.lambda_min) + self.lambda_min

    def denorm_R(self, v: torch.Tensor) -> torch.Tensor:
        """Desnormaliza la intensidad (puede ser positiva o negativa)"""
        return v * self.r_max

    def build_spectrum(self, pred_norm: torch.Tensor, wl_real_t: torch.Tensor) -> torch.Tensor:
        """Reconstrucción del espectro derivando sigma de lambda."""
        batch_size = pred_norm.shape[0]
        
        # Reshape a (Batch, Transiciones, 2 parámetros: lambda, R)
        pred_reshaped = pred_norm.view(batch_size, self.num_transiciones, 2)
        
        # 1. Desnormalización vectorial
        lambda_real = self.denorm_lambda(pred_reshaped[:, :, 0:1])
        r_real = self.denorm_R(pred_reshaped[:, :, 1:2])
        
        # 2. FÍSICA ACOPLADA: Cálculo de sigma dependiente de lambda
        # Al usar tensores de PyTorch, la derivada parcial d(Loss)/d(lambda) 
        # viajará también por aquí automáticamente.
        sigma_nm = ((lambda_real**2) / self.hc_ev_nm) * self.sigma_ev + self.epsilon
        
        # 3. Malla tridimensional (Broadcasting)
        wl_expanded = wl_real_t.view(1, 1, -1)
        
        # 4. Cálculo Gaussiano
        exponente = -((wl_expanded - lambda_real)**2) / (2 * (sigma_nm**2))
        gaussianas_matrix = r_real * torch.exp(exponente)
        
        return torch.sum(gaussianas_matrix, dim=1)

    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor, target_spec: torch.Tensor, wl_real_t: torch.Tensor) -> tuple:
        """
        Calcula el error comparando los 6 parámetros y el espectro final de 100 puntos.
        """
        # --- A. ERROR PARAMÉTRICO (MSE de los 6 parámetros) ---
        loss_params = self.mse(pred_norm, target_norm)
        
        # --- B. ERROR ESPECTROSCÓPICO (Física) ---
        pred_spec = self.build_spectrum(pred_norm, wl_real_t)
        
        # Normalizamos la pérdida espectral dividiendo por r_max para mantener escalas comparables
        loss_spectra = self.mse(pred_spec / self.r_max, target_spec / self.r_max)
        
        # --- C. HÍBRIDO ---
        loss_total = self.alpha * loss_params + self.beta * loss_spectra
        
        return loss_total, loss_params, loss_spectra
# =====================================================================
# --- PINN FASE B: TRANSICIONES (100 Transiciones) ---
# =====================================================================

class PINNLoss(nn.Module):
    """
    Función de pérdida informada por la física (PINN) vectorizada.
    Integra la desnormalización, la máscara dinámica ("Zombis") y la reconstrucción
    espectral en un único grafo computacional sin bucles explícitos.
    """
    def __init__(self, 
                 num_gaussianas: int,
                 a_max: float, 
                 mu_min: float, 
                 mu_max: float, 
                 sigma_max: float,
                 alpha: float = 1.0, 
                 beta: float = 1.0):
        super(PINNLoss, self).__init__()
        self.num_gaussianas = num_gaussianas
        self.a_max = a_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_max = sigma_max
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def denorm_A(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.a_max

    def denorm_mu(self, v: torch.Tensor) -> torch.Tensor:
        return (v + 1.0) / 2.0 * (self.mu_max - self.mu_min) + self.mu_min

    def denorm_sigma(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.sigma_max

    def build_spectrum(self, pred_norm: torch.Tensor, wl_real_t: torch.Tensor) -> torch.Tensor:
        """Reconstrucción tensorial vectorizada O(1)."""
        batch_size = pred_norm.shape[0]
        
        # Reshape a (Batch, N_Gaussianas, 3 parámetros)
        pred_reshaped = pred_norm.view(batch_size, self.num_gaussianas, 3)
        
        # Desnormalización vectorial
        A_real = self.denorm_A(pred_reshaped[:, :, 0:1])
        mu_real = self.denorm_mu(pred_reshaped[:, :, 1:2])
        sig_real = self.denorm_sigma(pred_reshaped[:, :, 2:3])
        sig_real = torch.clamp(sig_real, min=1e-4)
        
        # Expansión topológica para broadcasting: (Batch, N_Gauss, Puntos_X)
        wl_expanded = wl_real_t.view(1, 1, -1)
        
        # Grafo físico
        exponente = -0.5 * ((wl_expanded - mu_real) / sig_real)**2
        gaussianas_matrix = A_real * torch.exp(exponente)
        
        return torch.sum(gaussianas_matrix, dim=1)

    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor, target_spec: torch.Tensor, wl_real_t: torch.Tensor) -> tuple:
        batch_size = pred_norm.shape[0]
        
        # --- A. MÁSCARA DINÁMICA VECTORIZADA ---
        target_reshaped = target_norm.view(batch_size, self.num_gaussianas, 3)
        A_target = target_reshaped[:, :, 0]
        
        is_alive = (torch.abs(A_target) > 1e-4).float()
        mask = is_alive.unsqueeze(-1).expand(-1, -1, 3).reshape(batch_size, -1)
        
        # --- B. ERROR PARAMÉTRICO ---
        diff = pred_norm - target_norm
        masked_diff = diff * mask
        active_params = torch.clamp(torch.sum(mask), min=1.0)
        loss_params = torch.sum(masked_diff**2) / active_params
        
        # --- C. ERROR ESPECTROSCÓPICO ---
        pred_spec = self.build_spectrum(pred_norm, wl_real_t)
        loss_spectra = self.mse(pred_spec / self.a_max, target_spec / self.a_max)
        
        # --- D. HÍBRIDO ---
        loss_total = self.alpha * loss_params + self.beta * loss_spectra
        
        return loss_total, loss_params, loss_spectra
# =====================================================================
# --- CAJAS NEGRAS PURAS (Sin Física Acoplada) ---
# =====================================================================
class EndToEndLoss(nn.Module):
    """
    Pérdida para el modelo End-to-End.
    La red predice los 100 puntos del espectro directamente.
    """
    def __init__(self):
        super(EndToEndLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_norm: torch.Tensor, target_spec: torch.Tensor, wl_real_t: torch.Tensor) -> tuple:
        # La predicción se compara directamente con la curva espectral
        loss = self.mse(pred, target_spec)
        # Devolvemos 0.0 en los parámetros para mantener la compatibilidad con el engine.py
        return loss, torch.tensor(0.0, device=pred.device), loss


class ParametricLoss(nn.Module):
    """
    Pérdida para el modelo paramétrico de caja negra.
    La red predice los parámetros (ej. las 10 gaussianas), pero NO se dibuja 
    el espectro para retropropagar el error.
    """
    def __init__(self):
        super(ParametricLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_norm: torch.Tensor, target_spec: torch.Tensor, wl_real_t: torch.Tensor) -> tuple:
        # La predicción se compara solo con los parámetros objetivo
        loss = self.mse(pred, target_norm)
        # Devolvemos 0.0 en el espectro para mantener la compatibilidad
        return loss, loss, torch.tensor(0.0, device=pred.device)