"""
Módulo para análisis post-sesión de Rangos de Movimiento (ROM).
Detecta repeticiones, extrae máximos y mínimos, y calcula simetría.

CAMBIOS respecto a la versión original (ver "FIX"):
  1. Remuestreo a rejilla uniforme antes de Savitzky-Golay (los frames con
     angle=None generaban huecos y savgol asume muestreo uniforme).
  2. Mínimos por repetición con left_bases/right_bases de find_peaks.
  3. duration_s calculado (antes placeholder 0.0).
  4. Se reportan máximo/mínimo globales aunque no haya repeticiones.
  5. Simetría también sobre el ángulo máximo alcanzado.
  6. Prominencia y ventana configurables. Los ángulos que llegan ya son
     clínicos (0° = extendido), así que un pico = flexión máxima.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import List, Dict, Any, Optional


class ROMAnalyzer:
    def __init__(self, fps: float = 30.0, prominence: float = 15.0,
                 smooth_window: int = 15, smooth_poly: int = 3,
                 max_gap_s: float = 0.5, min_duration_s: float = 1.0):
        self.fps = fps
        self.prominence = prominence
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.max_gap_s = max_gap_s          # huecos mayores no se interpolan
        self.min_duration_s = min_duration_s

    # FIX 1
    def _resample_uniform(self, angles: np.ndarray, timestamps: np.ndarray):
        """Interpola a rejilla uniforme a self.fps. Marca como NaN los huecos
        mayores a max_gap_s para no inventar datos donde el joint no se vio."""
        order = np.argsort(timestamps)
        t, a = timestamps[order], angles[order]
        grid = np.arange(t[0], t[-1], 1.0 / self.fps)
        if len(grid) < 2:
            return grid, a[:len(grid)]
        resampled = np.interp(grid, t, a)
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        gap = t[idx] - t[idx - 1]
        resampled[gap > self.max_gap_s] = np.nan
        return grid, resampled

    def _smooth_signal(self, data: np.ndarray) -> np.ndarray:
        """Savitzky-Golay por tramos continuos (sin NaN)."""
        out = data.copy()
        valid = ~np.isnan(data)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(int), [0]))))
        for s, e in zip(edges[::2], edges[1::2]):
            seg = data[s:e]
            if len(seg) >= self.smooth_window:
                out[s:e] = savgol_filter(seg, self.smooth_window, self.smooth_poly)
        return out

    def detect_repetitions(self, angles: List[float], timestamps: List[float]) -> Dict[str, Any]:
        if len(angles) < 30:
            return {"num_reps": 0, "error": "Datos insuficientes"}

        arr = np.asarray(angles, dtype=float)
        ts = np.asarray(timestamps, dtype=float)
        grid, resampled = self._resample_uniform(arr, ts)
        if len(grid) < 2:
            return {"num_reps": 0, "error": "Datos insuficientes"}
        smoothed = self._smooth_signal(resampled)

        # FIX 4: estadísticas globales siempre disponibles (datos crudos válidos)
        overall = {
            "overall_max": round(float(np.nanmax(arr)), 1),
            "overall_min": round(float(np.nanmin(arr)), 1),
            "duration_recorded_s": round(float(ts.max() - ts.min()), 1),
        }

        filled = np.where(np.isnan(smoothed), np.nanmin(smoothed), smoothed)  # NaN no cuenta como pico
        distance = max(int(self.fps * 0.5), 1)
        peaks, props = find_peaks(filled, prominence=self.prominence, distance=distance)

        reps = []
        for i, p in enumerate(peaks):
            # FIX 2: bases de prominencia = valles reales a cada lado del pico
            lb, rb = int(props["left_bases"][i]), int(props["right_bases"][i])
            rep_max = float(smoothed[p])
            rep_min = float(min(filled[lb], filled[rb]))
            reps.append({
                "max_angle": round(rep_max, 1),
                "min_angle": round(rep_min, 1),
                "range": round(rep_max - rep_min, 1),
                "duration_s": round(float(grid[rb] - grid[lb]), 2),  # FIX 3
            })

        if not reps:
            return {"num_reps": 0, **overall}

        ranges = [r["range"] for r in reps]
        maxes = [r["max_angle"] for r in reps]
        return {
            "num_reps": len(reps),
            "avg_range": round(float(np.mean(ranges)), 1),
            "std_range": round(float(np.std(ranges)), 1),
            "avg_max_angle": round(float(np.mean(maxes)), 1),
            "repetitions": reps,
            **overall,
        }

    def calculate_symmetry(self, right_val: float, left_val: float) -> float:
        """SI = |D - I| / (0.5 * (D + I)) * 100. Cercano a 0 = simétrico."""
        denom = 0.5 * (right_val + left_val)
        if denom == 0:
            return 0.0
        return round(abs(right_val - left_val) / denom * 100.0, 1)

    def analyze_session(self, all_measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        joint_series: Dict[str, Dict[str, list]] = {}
        for record in all_measurements:
            if record.get("mode") != "body":
                continue
            ts = record.get("timestamp", 0.0)
            for joint_key, data in record.get("angles", {}).items():
                if data.get("angle") is None:
                    continue
                s = joint_series.setdefault(joint_key, {"angles": [], "timestamps": []})
                s["angles"].append(data["angle"])
                s["timestamps"].append(ts)

        report = {}
        for joint_key, series in joint_series.items():
            analysis = self.detect_repetitions(series["angles"], series["timestamps"])
            # FIX 4: se reporta la articulación aunque no haya repeticiones,
            # siempre que haya datos suficientes (overall_max presente)
            if analysis.get("num_reps", 0) > 0 or "overall_max" in analysis:
                report[joint_key] = analysis

        symmetry = {}
        pairs = [("hombro_der", "hombro_izq"), ("codo_der", "codo_izq"),
                 ("cadera_der", "cadera_izq"), ("rodilla_der", "rodilla_izq"),
                 ("muneca_der", "muneca_izq")]
        for r_key, l_key in pairs:
            if r_key in report and l_key in report:
                r, l = report[r_key], report[l_key]
                entry = {}
                if "avg_range" in r and "avg_range" in l:
                    entry["range_symmetry_index_pct"] = self.calculate_symmetry(r["avg_range"], l["avg_range"])
                    entry["right_range"], entry["left_range"] = r["avg_range"], l["avg_range"]
                # FIX 5: simetría sobre el ROM máximo alcanzado
                entry["max_angle_symmetry_index_pct"] = self.calculate_symmetry(r["overall_max"], l["overall_max"])
                entry["right_max"], entry["left_max"] = r["overall_max"], l["overall_max"]
                symmetry[f"{r_key}_vs_{l_key}"] = entry

        return {"version": "1.1", "joints_analyzed": report, "symmetry_analysis": symmetry}