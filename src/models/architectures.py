"""
Módulo de Arquitecturas de Deep Learning.
Contiene las definiciones de las redes neuronales parametrizables.
Diseñado para ser agnóstico a la dimensionalidad de las entradas (descriptores moleculares)
y de las salidas (espectro continuo o parámetros de Gaussianas).
Incluye la simetría C2 específica de los helicenos.
"""

import torch
import torch.nn as nn
from typing import List

# =====================================================================
# --- CLASE MAESTRA DE RED NEURONAL (Física + Optuna) ---
# =====================================================================
class DynamicPINN(nn.Module):
    """
    Arquitectura unificada tipo Perceptrón Multicapa (MLP) para predicción de Dicroísmo Circular.
    Integra el promediado de simetría (forward original y reverso) para la invarianza física (C2).
    
    Compatible dinámicamente con Optuna y con cualquier dimensionalidad:
    - End-to-End (Caja Negra): input_dim=16, output_dim=100
    - PINN (10 Gaussianas): input_dim=16, output_dim=30
    - Fase A (3 Transiciones): input_dim=16, output_dim=6
    """
    
    def __init__(self, input_dim=16, output_dim=30, n_layers=3, layer_0_size=256, **kwargs):
        super(DynamicPINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Construcción dinámica de la topología de la red (Para Optuna)
        layers = []
        in_features = input_dim
        
        for i in range(n_layers):
            # Extraemos el tamaño de la capa de Optuna, o usamos el por defecto
            out_features = kwargs.get(f'layer_{i}_size', layer_0_size)
            
            # 1. Capa Lineal
            layers.append(nn.Linear(in_features, out_features))
            
            # 2. Función de Activación (GELU, superior para regresión continua)
            layers.append(nn.GELU())
            
            # 3. Regularización Suave: Replicamos tu diseño original
            # Dropout (10%) SOLO en la primera capa para evitar sobreajuste a posiciones concretas
            if i == 0:
                layers.append(nn.Dropout(0.1))
                
            in_features = out_features
            
        # Capa de salida (Libre sin Tanh, el rango se controla en la Loss y en el Escalador)
        layers.append(nn.Linear(in_features, output_dim))
        
        # Empaquetamos todas las capas en un módulo secuencial
        self.net = nn.Sequential(*layers)
        
        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Inicializa los pesos de la red para evitar gradientes desvanecientes."""
        if isinstance(m, nn.Linear):
            # Usamos inicialización de Kaiming (He) ideal para redes profundas con GELU/ReLU
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red.
        Aplica el truco de Simetría Molecular: Promedia la predicción de la entrada 
        original y su reverso (simetría C2 del heliceno) para reducir varianza.
        """
        x_flip = torch.flip(x, dims=[1])
        return 0.5 * (self.net(x) + self.net(x_flip))

# =====================================================================
# --- PRUEBA DE INTEGRIDAD (Ejecutable) ---
# =====================================================================
if __name__ == "__main__":
    # Prueba 1: Fase A (Predicción de 3 Transiciones discretas)
    # Entrada: 16 posiciones Hammett. Salida: 6 (3 Amplitudes, 3 Posiciones)
    modelo_fase_a = DynamicPINN(input_dim=16, output_dim=6, n_layers=2, layer_0_size=128, layer_1_size=128)
    tensor_prueba = torch.randn(32, 16) # Simulamos un batch de 32 moléculas con 16 posiciones
    salida_a = modelo_fase_a(tensor_prueba)
    print(f"✅ Prueba Fase A (3T): Salida esperada [32, 6] -> Obtenida {list(salida_a.shape)}")
    
    # Prueba 2: Fase B (PINN para 10 Gaussianas)
    # Entrada: 16 posiciones Hammett. Salida: 30 (10 Amplitudes, 10 mu, 10 sigma)
    modelo_fase_b = DynamicPINN(input_dim=16, output_dim=30, n_layers=3, layer_0_size=256)
    salida_b = modelo_fase_b(tensor_prueba)
    print(f"✅ Prueba Fase B (PINN 10G): Salida esperada [32, 30] -> Obtenida {list(salida_b.shape)}")
    
    # Prueba 3: Modelo End-to-End (Directo a los 100 puntos - Caja Negra)
    # Entrada: 16 posiciones Hammett. Salida: 100 puntos del espectro continuo
    modelo_e2e = DynamicPINN(input_dim=16, output_dim=100, n_layers=4)
    salida_c = modelo_e2e(tensor_prueba)
    print(f"✅ Prueba End-to-End: Salida esperada [32, 100] -> Obtenida {list(salida_c.shape)}")