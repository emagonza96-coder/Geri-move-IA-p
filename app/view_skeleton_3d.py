"""
Visualizador 3D de Esqueleto desde Sesión Grabada
==================================================
Lee un JSON generado por main.py y muestra el esqueleto
animado en 3D usando matplotlib.

Uso:
    python app/view_skeleton_3d.py output/sesion_XXXX_data.json

Controles:
    - La animación corre automáticamente.
    - Arrastra con el mouse para rotar la vista 3D.
    - Cierra la ventana para salir.
"""

import json
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ──────────────────────────────────────────────
# Conexiones del esqueleto de MediaPipe (33 puntos)
# ──────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Brazo izquierdo
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # Brazo derecho
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Pierna izquierda
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Pierna derecha
    (24, 26), (26, 28), (28, 30), (28, 32),
    # Cara
    (0, 11), (0, 12),
]

# Colores por segmento corporal
SEGMENT_COLORS = {
    "torso":   "#00C9B1",
    "brazo_i": "#4FC3F7",
    "brazo_d": "#F48FB1",
    "pierna_i":"#81C784",
    "pierna_d":"#FFB74D",
    "cara":    "#B0BEC5",
}

CONNECTION_COLORS = [
    SEGMENT_COLORS["torso"],    # 11-12
    SEGMENT_COLORS["torso"],    # 11-23
    SEGMENT_COLORS["torso"],    # 12-24
    SEGMENT_COLORS["torso"],    # 23-24
    SEGMENT_COLORS["brazo_i"],  # 11-13
    SEGMENT_COLORS["brazo_i"],  # 13-15
    SEGMENT_COLORS["brazo_i"],  # 15-17
    SEGMENT_COLORS["brazo_i"],  # 15-19
    SEGMENT_COLORS["brazo_i"],  # 15-21
    SEGMENT_COLORS["brazo_d"],  # 12-14
    SEGMENT_COLORS["brazo_d"],  # 14-16
    SEGMENT_COLORS["brazo_d"],  # 16-18
    SEGMENT_COLORS["brazo_d"],  # 16-20
    SEGMENT_COLORS["brazo_d"],  # 16-22
    SEGMENT_COLORS["pierna_i"], # 23-25
    SEGMENT_COLORS["pierna_i"], # 25-27
    SEGMENT_COLORS["pierna_i"], # 27-29
    SEGMENT_COLORS["pierna_i"], # 27-31
    SEGMENT_COLORS["pierna_d"], # 24-26
    SEGMENT_COLORS["pierna_d"], # 26-28
    SEGMENT_COLORS["pierna_d"], # 28-30
    SEGMENT_COLORS["pierna_d"], # 28-32
    SEGMENT_COLORS["cara"],     # 0-11
    SEGMENT_COLORS["cara"],     # 0-12
]


def load_landmarks_from_json(filepath: str):
    """
    Carga los frames de landmarks desde el JSON de sesión.
    Retorna lista de arrays numpy con forma (33, 3).
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    frames = []
    frame_angles = []
    
    for measurement in data.get("measurements", []):
        lms = measurement.get("landmarks")
        if not lms or len(lms) < 33:
            continue

        coords = np.array([
            [lm["x"], lm["z"], -lm["y"]]   # Ajuste de ejes: z = profundidad, y invertida = altura
            for lm in sorted(lms, key=lambda l: l["id"])
        ])
        
        # Filtrar landmarks con baja visibilidad (reemplazar con NaN para no dibujar)
        visibility = np.array([lm["visibility"] for lm in sorted(lms, key=lambda l: l["id"])])
        coords[visibility < 0.3] = np.nan
        
        frames.append(coords)
        frame_angles.append(measurement.get("angles", {}))
        
    return frames, frame_angles, data


def build_angle_text(angles: dict) -> str:
    lines = []
    for key, val in angles.items():
        if val.get("angle") is not None:
            lines.append(f"{key:12s}: {val['angle']:6.1f}°")
    return "\n".join(lines) if lines else "Sin ángulos"


def main():
    parser = argparse.ArgumentParser(description="Visualizador 3D de Esqueleto")
    parser.add_argument("json_file", help="Archivo JSON de sesión (output/*.json)")
    parser.add_argument("--speed", type=float, default=1.0, help="Multiplicador de velocidad (default: 1.0)")
    parser.add_argument("--skip", type=int, default=1, help="Mostrar 1 de cada N frames (default: 1)")
    args = parser.parse_args()

    print(f"[INFO] Cargando sesión: {args.json_file}")
    frames, frame_angles, meta = load_landmarks_from_json(args.json_file)
    
    if not frames:
        print("[ERROR] No se encontraron landmarks en el archivo JSON.")
        print("[HINT]  Graba una nueva sesión con la versión actualizada de main.py.")
        sys.exit(1)

    # Submuestrear frames si se especifica --skip
    frames = frames[::args.skip]
    frame_angles = frame_angles[::args.skip]
    
    n_frames = len(frames)
    fps_original = meta.get("fps", 30.0)
    interval_ms = int((1000.0 / fps_original) / args.speed) * args.skip
    
    print(f"[INFO] {n_frames} frames cargados. Sesión: {meta.get('session_id', 'desconocida')}")
    print(f"[INFO] Resolución original: {meta.get('resolution', 'N/A')}")
    print("[INFO] Abre la ventana 3D. Arrastra para rotar la vista.")

    # ── Configurar figura oscura ──────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8), facecolor="#0D0D0D")
    fig.suptitle("Mobility Scan — Visualizador 3D de Esqueleto", 
                 color="#00C9B1", fontsize=14, fontweight="bold")

    # Subplot 3D del esqueleto
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.set_facecolor("#0D0D0D")
    ax3d.set_title("Esqueleto 3D (MediaPipe Z)", color="white", fontsize=10)
    
    # Subplot 2D de ángulos en tiempo real
    ax_angles = fig.add_subplot(122)
    ax_angles.set_facecolor("#0D0D0D")
    ax_angles.axis("off")
    angle_text_obj = ax_angles.text(
        0.05, 0.95, "", transform=ax_angles.transAxes,
        color="#00C9B1", fontsize=11, verticalalignment="top",
        fontfamily="monospace"
    )
    ax_angles.set_title("Ángulos Articulares", color="white", fontsize=10)

    # Calcular rango global de coordenadas para escala fija
    all_coords = np.concatenate(frames, axis=0)
    valid = all_coords[~np.isnan(all_coords).any(axis=1)]
    margin = 0.1
    x_range = [valid[:, 0].min() - margin, valid[:, 0].max() + margin]
    y_range = [valid[:, 1].min() - margin, valid[:, 1].max() + margin]
    z_range = [valid[:, 2].min() - margin, valid[:, 2].max() + margin]

    # Objetos de dibujo iniciales
    scatter = ax3d.scatter([], [], [], c="#FFFFFF", s=25, zorder=5)
    lines = [ax3d.plot([], [], [], c=c, linewidth=2)[0] for c in CONNECTION_COLORS]
    frame_counter = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes, color="gray", fontsize=8)

    def setup_axes():
        ax3d.set_xlim(x_range)
        ax3d.set_ylim(y_range)
        ax3d.set_zlim(z_range)
        ax3d.set_xlabel("X", color="gray", labelpad=1)
        ax3d.set_ylabel("Z (Prof.)", color="gray", labelpad=1)
        ax3d.set_zlabel("Altura", color="gray", labelpad=1)
        ax3d.tick_params(colors="gray", labelsize=7)
        for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#333333")
        ax3d.grid(True, color="#1A1A1A")

    setup_axes()

    def update(frame_idx):
        coords = frames[frame_idx]   # (33, 3)
        angles = frame_angles[frame_idx]

        # Actualizar puntos
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        scatter._offsets3d = (xs, ys, zs)

        # Actualizar huesos
        for i, (a, b) in enumerate(SKELETON_CONNECTIONS):
            pa, pb = coords[a], coords[b]
            if np.isnan(pa).any() or np.isnan(pb).any():
                lines[i].set_data([], [])
                lines[i].set_3d_properties([])
            else:
                lines[i].set_data([pa[0], pb[0]], [pa[1], pb[1]])
                lines[i].set_3d_properties([pa[2], pb[2]])

        # Ángulos en texto
        angle_text_obj.set_text(build_angle_text(angles))
        frame_counter.set_text(f"Frame {frame_idx + 1}/{n_frames}")
        
        return [scatter, *lines, angle_text_obj, frame_counter]

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True
    )

    plt.tight_layout()
    plt.show()
    print("[INFO] Visualización cerrada.")


if __name__ == "__main__":
    main()
