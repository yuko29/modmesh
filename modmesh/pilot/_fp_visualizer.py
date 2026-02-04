from ._gui_common import PilotFeature

from ._base_app import PlotArea

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDockWidget, QWidget)

from dataclasses import dataclass
import numpy as np

__all__ = [  # noqa: F822
    'FPVisualizer',
]


class FPVisualizer(PilotFeature):

    def populate_menu(self):
        self._add_menu_item(
            menu=self._mgr.visualizeMenu,
            text="Visualize FP8 Data",
            tip="Visualize FP8 Data",
            func=self.run,
        )

    def run(self):
        self.setup()

        self._mgr.mainWindow.addDockWidget(
                Qt.LeftDockWidgetArea,
                QDockWidget()
            )

        # A new sub-window (`QMdiSubWindow`) for the plot area
        self._subwin = self._mgr.addSubWindow(QWidget())
        self._subwin.setWidget(PlotArea(self))
        self._subwin.showMaximized()
        self._subwin.show()

    def setup(self):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = canvas.figure.subplots()
        fig.tight_layout()
        data = generate_fp8_values(
            FP8Format(
                exp_bits=4, mant_bits=3, exponent_bias=7
            )
        )
        ax.scatter(data, 1 * np.ones_like(data), alpha=0.5)

        self.plot = canvas


@dataclass
class FP8Format:
    exp_bits: int
    mant_bits: int
    exponent_bias: int = None


def fp8_to_float(bits, fmt: FP8Format):
    """Convert an 8-bit floating point representation to float32"""
    sign = (bits >> 7) & 0x1
    exponent = (bits >> fmt.mant_bits) & ((1 << fmt.exp_bits) - 1)
    mantissa = bits & ((1 << fmt.mant_bits) - 1)

    if exponent == (1 << fmt.exp_bits) - 1:  # Inf/NaN
        if mantissa != 0:
            return np.nan
        else:
            return np.inf if sign == 0 else -np.inf
    elif exponent == 0:  # Subnormal or zero
        if mantissa == 0:
            return 0.0
        else:
            # Subnormal number
            exp_val = 1 - ((1 << (fmt.exp_bits - 1)) - 1)
            mant_val = mantissa / (1 << fmt.mant_bits)
    else:
        # Normalized number
        exp_val = exponent - fmt.exponent_bias
        mant_val = 1 + (mantissa / (1 << fmt.mant_bits))

    value = mant_val * (2 ** exp_val)
    return value if sign == 0 else -value


def generate_fp8_values(fmt: FP8Format):
    """Generate all positive finite fp8 values for a given format"""
    values = []

    for bits in range(256):
        value = fp8_to_float(bits, fmt)
        if np.isfinite(value) and value > 0:
            values.append(value)

    return np.array(values)
