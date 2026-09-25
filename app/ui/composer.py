"""
Compositor de superficies para la ventana principal de Mobility Scan.

Combina la superficie de video y la superficie del panel en un unico
array numpy para cv2.imshow. Cada superficie mantiene su resolucion
interna; solo se escala el panel verticalmente para coincidir con el
alto del video.
"""

import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DisplayComposer:
    """Combina la superficie de video y la del panel para mostrar en pantalla."""

    def compose(
        self,
        video_surface: np.ndarray,
        panel_surface: np.ndarray,
    ) -> np.ndarray:
        """
        Combina video y panel horizontalmente, igualando alturas.

        El video se muestra a su resolucion nativa. El panel se escala
        verticalmente para coincidir con el alto del video, preservando
        la proporcion de su ancho.

        Args:
            video_surface: Frame de video BGR (con o sin overlay de esqueleto).
            panel_surface: Panel renderizado de forma independiente (BGR).

        Returns:
            Array numpy BGR combinado, listo para cv2.imshow.
        """
        vh, vw = video_surface.shape[:2]
        ph, pw = panel_surface.shape[:2]

        target_h = vh

        if ph != target_h:
            scale = target_h / ph
            new_pw = max(1, int(pw * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            panel_scaled = cv2.resize(
                panel_surface, (new_pw, target_h), interpolation=interp
            )
        else:
            panel_scaled = panel_surface

        return np.hstack([video_surface, panel_scaled])
