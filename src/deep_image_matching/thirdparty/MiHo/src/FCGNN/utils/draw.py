import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def draw_matches(img1, img2, matches, color='g', filename='matches.png'):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=img1.dtype)

    canvas[:h1, :w1] = img1[:,:,np.newaxis]
    canvas[:h2, w1:] = img2[:,:,np.newaxis]

    plt.figure(figsize=(15,5))
    plt.axis("off")
    plt.imshow(canvas, zorder=1)

    xs = matches[:, [0, 2]]
    xs[:, 1] += w1
    ys = matches[:, [1, 3]]

    plt.plot(
        xs.T, ys.T,
        alpha=1,
        linestyle="-",
        linewidth=0.5,
        aa=False,
        marker='o',
        markersize=2,
        fillstyle='none',
        color=color,
        zorder=2,
    )

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
