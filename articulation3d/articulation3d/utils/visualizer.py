import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure

from detectron2.utils.visualizer import Visualizer


class ArtiVisualizer(Visualizer):
    """
    ArtiVisualizer extends detectron2 Visualizer and allow us to draw axes.
    """

    def draw_arrow(self, x_data, y_data, color, linestyle="-", linewidth=None):
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)

        self.output.ax.arrow(
            x=x_data[0],
            y=y_data[0],
            dx=(x_data[1] - x_data[0]) * 1.0,
            dy=(y_data[1] - y_data[0]) * 1.0,
            width=linewidth * self.output.scale,
            head_width=linewidth * self.output.scale * 5.0,
            length_includes_head=True,
            color=color,
            overhang=0.5,
            linestyle=linestyle,
        )

        return self.output
