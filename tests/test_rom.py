import numpy as np
import pytest
from app.biomechanics.rom_analyzer import ROMAnalyzer

def test_rom_analyzer_detect_repetitions():
    """Prueba que el analizador detecta correctamente repeticiones en una señal senoidal sintética."""
    fps = 30.0
    duration = 5.0 # segundos
    t = np.linspace(0, duration, int(fps * duration))
    
    # Crear una señal base en 0
    angles = np.zeros_like(t)
    # Rellenar el centro con 3 ciclos (de t=1 a t=4.33, freq=0.9)
    freq = 0.9
    cycle_time = 1.0 / freq
    start_idx = int(1.0 * fps)
    end_idx = start_idx + int(3 * cycle_time * fps)
    
    t_cycles = t[start_idx:end_idx] - t[start_idx]
    angles[start_idx:end_idx] = -45 * np.cos(2 * np.pi * freq * t_cycles) + 45
    timestamps = list(t)
    
    # Agregar algo de ruido que no debería afectar el conteo por la prominencia
    noise = np.random.normal(0, 2, len(t))
    noisy_angles = angles + noise
    
    analyzer = ROMAnalyzer(fps=fps)
    result = analyzer.detect_repetitions(list(noisy_angles), timestamps)
    
    assert result["num_reps"] == 3, f"Se esperaban 3 repeticiones, detectadas {result['num_reps']}"
    assert 85 <= result["avg_max_angle"] <= 95, f"Rango máximo esperado ~90, obtenido {result['avg_max_angle']}"
    assert 85 <= result["avg_range"] <= 95, f"Rango total esperado ~90, obtenido {result['avg_range']}"
    
def test_rom_analyzer_symmetry():
    analyzer = ROMAnalyzer()
    
    # Perfect symmetry
    assert analyzer.calculate_symmetry(100.0, 100.0) == 0.0
    
    # D: 100, I: 50 -> dif 50. denom: 75 -> 50/75 * 100 = 66.6...
    assert 66.6 <= analyzer.calculate_symmetry(100.0, 50.0) <= 66.7
