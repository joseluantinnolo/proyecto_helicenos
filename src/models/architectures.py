"""
Módulo de Arquitecturas de Deep Learning.
Contiene las definiciones de las redes neuronales parametrizables.
Diseñado para ser agnóstico a la dimensionalidad de las entradas (descriptores moleculares)
y de las salidas (espectro continuo o parámetros de Gaussianas).
"""

import torch
import torch.nn as nn
from typing import List

# =====================================================================
# --- CLASE MAESTRA DE RED NEURONAL ---
# =====================================================================
class SpectraPredictorNN(nn.Module):
    """
    Arquitectura unificada tipo Perceptrón Multicapa (MLP) para predicción de Dicroísmo Circular.
    
    Permite instanciar redes dinámicas. Ejemplos de uso:
    - End-to-End (3 trans): input_dim=16, output_dim=100
    - PINN (10 Gaussianas): input_dim=16, output_dim=30 (10x amp, 10x mu, 10x sigma)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_layers: List[int] = [256, 512, 256], 
        dropout_rate: float = 0.2,
        use_batchnorm: bool = True
    ):
        super(SpectraPredictorNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Construcción dinámica de la topología de la red
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_layers:
            # 1. Capa Lineal (Pesos y Sesgos)
            layers.append(nn.Linear(current_dim, h_dim))
            
            # 2. Normalización por Lotes
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
                
            # 3. Función de Activación (GELU)
            layers.append(nn.GELU())
            
            # 4. Regularización
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
                
            current_dim = h_dim
            
        # Capa de salida (Sin activación para permitir regresión en rango real)
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Empaquetamos todas las capas en un módulo secuencial
        self.network = nn.Sequential(*layers)
        
        # Inicialización de pesos de He (Kaiming)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Inicializa los pesos de la red para evitar gradientes desvanecientes."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red.
        """
        return self.network(x)

# =====================================================================
# --- PRUEBA DE INTEGRIDAD (Ejecutable) ---
# =====================================================================
if __name__ == "__main__":
    # Prueba 1: Fase A (Predicción de 6 parámetros discretos)
    # Entrada: 16 posiciones Hammett. Salida: 6 (3 lambda, 3 R)
    modelo_fase_a = SpectraPredictorNN(input_dim=16, output_dim=6, hidden_layers=[128, 128])
    tensor_prueba = torch.randn(32, 16) # Simulamos un batch de 32 moléculas con 16 posiciones
    salida_a = modelo_fase_a(tensor_prueba)
    print(f"✅ Prueba Fase A (Parámetros): Salida esperada [32, 6] -> Obtenida {list(salida_a.shape)}")
    
    # Prueba 2: Fase B (PINN para 10 Gaussianas)
    # Entrada: 16 posiciones Hammett. Salida: 30 (10 Amplitudes, 10 mu, 10 sigma)
    modelo_fase_b = SpectraPredictorNN(input_dim=16, output_dim=30, hidden_layers=[512, 1024, 512])
    salida_b = modelo_fase_b(tensor_prueba)
    print(f"✅ Prueba Fase B (PINN 10 Gauss): Salida esperada [32, 30] -> Obtenida {list(salida_b.shape)}")
    
    # Prueba 3: Modelo End-to-End (Directo a los 100 puntos)
    # Entrada: 16 posiciones Hammett. Salida: 100 puntos del espectro continuo
    modelo_e2e = SpectraPredictorNN(input_dim=16, output_dim=100)
    salida_c = modelo_e2e(tensor_prueba)
    print(f"✅ Prueba End-to-End: Salida esperada [32, 100] -> Obtenida {list(salida_c.shape)}")