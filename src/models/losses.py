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
    La anchura (sigma) es derivada analíticamente de la longitud de onda (lambda).
    """
    def __init__(self, num_transiciones: int, alpha: float = 1.0, beta: float = 1.0, 
                 a_max: float = 100.0, mu_min: float = 150.0, mu_max: float = 650.0, 
                 sigma_ev: float = 0.2, **kwargs):
        super(DiscretePINNLoss, self).__init__()
        self.num_transiciones = num_transiciones
        self.alpha = alpha
        self.beta = beta
        
        # Constantes del Pasaporte Físico
        self.a_max = a_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        
        self.sigma_ev = sigma_ev
        self.hc_ev_nm = 1240.0
        self.epsilon = 1e-5
        self.mse = nn.MSELoss()

    def denorm_lambda(self, v: torch.Tensor) -> torch.Tensor:
        """Devuelve la posición al rango real en nm para poder calcular la gaussiana."""
        return (v + 1.0) / 2.0 * (self.mu_max - self.mu_min) + self.mu_min

    def build_spectrum(self, pred_norm: torch.Tensor, wl_real_t: torch.Tensor) -> torch.Tensor:
        batch_size = pred_norm.shape[0]
        pred_reshaped = pred_norm.view(batch_size, self.num_transiciones, 2)
        
        # OJO AL ORDEN (Según tu DatasetDefinitivo): Col 0 es Lambda(nm), Col 1 es Amplitud(R)
        lambda_real = self.denorm_lambda(pred_reshaped[:, :, 0:1])
        r_norm = pred_reshaped[:, :, 1:2] # ¡La Amplitud se queda normalizada!
        
        # FÍSICA ACOPLADA: Cálculo de sigma dependiente de lambda real
        sigma_nm = ((lambda_real**2) / self.hc_ev_nm) * self.sigma_ev + self.epsilon
        
        wl_expanded = wl_real_t.view(1, 1, -1)
        
        exponente = -((wl_expanded - lambda_real)**2) / (2 * (sigma_nm**2))
        gaussianas_matrix = r_norm * torch.exp(exponente)
        
        return torch.sum(gaussianas_matrix, dim=1)

    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor, target_spec: torch.Tensor, wl_real_t: torch.Tensor) -> tuple:
        loss_params = self.mse(pred_norm, target_norm)
        
        pred_spec = self.build_spectrum(pred_norm, wl_real_t)
        
        # Ambos espectros están ahora normalizados en [-1, 1], el MSE es justo y estable
        loss_spectra = self.mse(pred_spec, target_spec)
        
        loss_total = self.alpha * loss_params + self.beta * loss_spectra
        return loss_total, loss_params, loss_spectra

# =====================================================================
# --- PINN FASE B: TRANSICIONES (8/10 Gaussianas) ---
# =====================================================================
class PINNLoss(nn.Module):
    """
    Función de pérdida informada por la física (PINN) vectorizada.
    Integra la reconstrucción Semi-Desescalada y la máscara "Zombis".
    """
    def __init__(self, num_gaussianas: int, alpha: float = 1.0, beta: float = 1.0,
                 a_max: float = 100.0, mu_min: float = 150.0, mu_max: float = 650.0, 
                 sigma_max: float = 60.0, **kwargs):
        super(PINNLoss, self).__init__()
        self.num_gaussianas = num_gaussianas
        self.alpha = alpha
        self.beta = beta
        
        # Constantes del Pasaporte Físico
        self.a_max = a_max
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma_max = sigma_max
        self.mse = nn.MSELoss()

    def denorm_mu(self, v: torch.Tensor) -> torch.Tensor:
        return (v + 1.0) / 2.0 * (self.mu_max - self.mu_min) + self.mu_min

    def denorm_sigma(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.sigma_max

    def build_spectrum(self, pred_norm: torch.Tensor, wl_real_t: torch.Tensor) -> torch.Tensor:
        batch_size = pred_norm.shape[0]
        pred_reshaped = pred_norm.view(batch_size, self.num_gaussianas, 3)
        
        # Orden Extractores (LeastSquares/PCA): [A, mu, sigma]
        A_norm = pred_reshaped[:, :, 0:1] # ¡La Amplitud se queda normalizada!
        mu_real = self.denorm_mu(pred_reshaped[:, :, 1:2])
        sig_real = self.denorm_sigma(pred_reshaped[:, :, 2:3])
        sig_real = torch.clamp(sig_real, min=1e-4)
        
        wl_expanded = wl_real_t.view(1, 1, -1)
        exponente = -0.5 * ((wl_expanded - mu_real) / sig_real)**2
        gaussianas_matrix = A_norm * torch.exp(exponente)
        
        return torch.sum(gaussianas_matrix, dim=1)

    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor, target_spec: torch.Tensor, wl_real_t: torch.Tensor) -> tuple:
        batch_size = pred_norm.shape[0]
        target_reshaped = target_norm.view(batch_size, self.num_gaussianas, 3)
        
        A_target_norm = target_reshaped[:, :, 0] 
        
        # Máscara zombi (con A normalizada, 1e-4 representa 0.01% de la altura máxima)
        is_alive = (torch.abs(A_target_norm) > 1e-4).float()
        mask = is_alive.unsqueeze(-1).expand(-1, -1, 3).reshape(batch_size, -1)
        
        diff = pred_norm - target_norm
        masked_diff = diff * mask
        active_params = torch.clamp(torch.sum(mask), min=1.0)
        loss_params = torch.sum(masked_diff**2) / active_params
        
        pred_spec = self.build_spectrum(pred_norm, wl_real_t)
        
        # Ambos espectros están ahora normalizados en [-1, 1]
        loss_spectra = self.mse(pred_spec, target_spec)
        
        loss_total = self.alpha * loss_params + self.beta * loss_spectra
        return loss_total, loss_params, loss_spectra

# =====================================================================
# --- CAJAS NEGRAS PURAS (Sin Física Acoplada) ---
# =====================================================================
class EndToEndLoss(nn.Module):
    def __init__(self, **kwargs):
        super(EndToEndLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_norm: torch.Tensor, target_spec: torch.Tensor, wl_real_t: torch.Tensor) -> tuple:
        loss = self.mse(pred, target_spec)
        return loss, torch.tensor(0.0, device=pred.device), loss

class ParametricLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ParametricLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_norm: torch.Tensor, target_spec: torch.Tensor, wl_real_t: torch.Tensor) -> tuple:
        loss = self.mse(pred, target_norm)
        return loss, loss, torch.tensor(0.0, device=pred.device)