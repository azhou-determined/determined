import logging
import warnings
from typing import Any

from determined.keras import callbacks

logger = logging.getLogger("determined.keras")


class TFKerasTensorBoard(callbacks.TensorBoard):
    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            "det.keras.TFKerasTensorBoard is a deprecated name for "
            "det.keras.callbacks.TensorBoard, please update your code.",
            FutureWarning,
            stacklevel=2,
        )
        # Avoid using super() due to a diamond inheritance pattern.
        callbacks.TensorBoard.__init__(self, *args, **kwargs)
