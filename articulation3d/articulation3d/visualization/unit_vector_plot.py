from qutip import Bloch
from qutip.states import basis
import numpy as np
import cv2
import matplotlib.pyplot as plt


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis = 2)
    return buf

def get_normal_figure(normal, history_normals=[], output_size=(480,640)):
    b = Bloch()
    if len(normal.numpy().shape) == 2:
        for n in normal:
            b.add_vectors(list(n.numpy()))
    else:
        if len(normal) != 0:
            normal = list(normal.numpy())
            b.add_vectors(normal)
    # print(history_normals)
    if len(history_normals) > 0:
        for hn in history_normals:
            # b.add_points(history_normals)
            b.add_points(hn)
    # import pdb;pdb.set_trace()
    b.zlabel = ['$z$','']
    b.ylabel = ['','$-y$']
    b.view = [-200, 30]

    b.render(b.fig, b.axes)
    img = fig2data(b.fig)
    plt.close(b.fig)

    ht, wd = img.shape[:2]
    resize_side = min(output_size[0], output_size[1], ht, wd)
    img = cv2.resize(img, (resize_side, resize_side))
    # create new image of desired size and color (blue) for padding
    result = np.full((output_size[0],output_size[1],3), (255,255,255), dtype=np.uint8)

    # compute center offset
    xx = (output_size[1] - resize_side) // 2
    yy = (output_size[0] - resize_side) // 2

    # copy img image into center of result image
    result[yy:yy+resize_side, xx:xx+resize_side] = img[:,:,:3]
    return result


def main():
    cv2.imwrite("debug/sphere.png", get_figure([0.2159, 0.8909, 0.3995]))


if __name__=='__main__':
    main()