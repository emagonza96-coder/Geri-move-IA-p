import yaml
from pathlib import Path
from typing import Dict, Any

_config_cache: Dict[str, Any] = {}

def load_config() -> Dict[str, Any]:
    """Carga la configuración centralizada desde settings.yaml (cacheada)."""
    global _config_cache
    if not _config_cache:
        config_path = Path(__file__).parent / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache

def get_rom(joint_key: str) -> tuple[float, float]:
    """Retorna el rango esperado de movimiento para una articulación específica."""
    config = load_config()
    rom_data = config.get("clinical_rom", {}).get(joint_key, {})
    rom = rom_data.get("expected_range")
    if rom and len(rom) == 2:
        return float(rom[0]), float(rom[1])
    return 0.0, 0.0

def get_clinical_details(joint_key: str) -> dict:
    """Retorna detalles adicionales (vista, plano, warnings) de una articulación."""
    config = load_config()
    return config.get("clinical_rom", {}).get(joint_key, {})

def get_threshold(key: str, default: float) -> float:
    """Obtiene un umbral de configuración."""
    config = load_config()
    return float(config.get("thresholds", {}).get(key, default))

def get_framing_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("framing", {})

def get_quality_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("quality", {})

def get_ui_config() -> Dict[str, Any]:
    config = load_config()
    return config.get("ui", {})
