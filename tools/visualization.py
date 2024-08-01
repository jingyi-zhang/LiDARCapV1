# -*- coding: utf-8 -*-=

import os, cv2, matplotlib, torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def read_image(path):
    assert os.path.exists(path), 'Not found %s' % path
    image = cv2.imread(path)
    if len(image.shape) == 3:
        image = image[:, :, ::-1]
    return image


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        dpi: dpi
        pad: pad
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


# noinspection PyUnresolvedReferences
def plot_matches(kpts0, kpts1, lc=None, lw=1.5, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0: corresponding keypoints of size (N, 2).
        kpts1: corresponding keypoints of size (N, 2).
        lc: line color
        lw: width of the lines.
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if lc is None:
        lc = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(lc) > 0 and not isinstance(lc[0], (tuple, list)):
        lc = [lc] * len(kpts0)
    
    if not isinstance(a, (tuple, list)):
        a = [a] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=lc[i], linewidth=lw,
            alpha=a[i])
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)
    fig.canvas.draw()


# noinspection PyUnresolvedReferences
def plot_keypoints(kpts0, kpts1, pc=None, ps=4, indices=(0, 1)):
    """Plot keypoints for a pair of existing images.
    Args:
        kpts0: corresponding keypoints of size (N, 2).
        kpts1: corresponding keypoints of size (N, 2).
        pc: match point color of each match, string or RGB tuple. Random if not given.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    # assert len(kpts0) == len(kpts1)
    if pc is None and len(kpts0) == len(kpts1):
        pc0 = pc1 = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(pc) > 0 and not isinstance(pc[0], (tuple, list)):
        if len(kpts0) == len(kpts1):
            pc0 = pc1 = [pc] * len(kpts0)
        else:
            pc0 = [pc] * len(kpts0)
            pc1 = [pc] * len(kpts1)
    else:
        pc0 = pc1 = pc

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=pc0, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=pc1, s=ps)
        # ax0.scatter(kpts0[:, 0], kpts0[:, 1], edgecolors=pc, s=ps, alpha=a)
        # ax1.scatter(kpts1[:, 0], kpts1[:, 1], edgecolors=pc, s=ps, alpha=a)
    
    fig.canvas.draw()


def emfactor(x, f=10, ceil=1.):
    assert 0 < ceil <= 1., 'ceill must in (0, 1])'
    x = (x - max(x)) * f
    x = np.exp(x)
    x = x*ceil
    return x


def fig2img(fig):
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    return image_from_plot


def img2tensor(x):
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    assert x.ndim == 3
    x = torch.tensor(x.copy()).float()
    if x.shape[2] == 3 or x.shape[2] == 1:
        x = x.permute(2, 0, 1)
    x = x / 255
    return x


gray = lambda x: np.float32(0.299) * x[:, :, 0] +\
                 np.float32(0.587) * x[:, :, 1] +\
                 np.float32(0.114) * x[:, :, 2]


def standardize_image(img):
    assert img.shape == 1 or img.shape == 3
    if img.shape[-1] == 1:
        return img[:, :, 0]
    return gray(img)


def train_visual(visual):
    img1, img2 = visual['img1'], visual['img2']
    keypoints0, keypoints1 = visual['keypoints0'], visual['keypoints1']
    matchkpts0, matchkpts1 = visual['matchkpts0'], visual['matchkpts1']
    
    plot_images([img1, img2], dpi=115)
    plot_keypoints(keypoints0, keypoints1, pc='#00ff00')
    plot_matches(matchkpts0, matchkpts1, lc='#00ff00', lw=0.5, a=1.)
    img = fig2img(plt.gcf())
    plt.close('all')
    return img


def epoch_visual(vbatch, path):
    imgs = [train_visual(v) for v in vbatch]
    imgs = [torch.from_numpy(np.copy(img)) for img in imgs]
    imgs = torch.stack(imgs).permute(0, 3, 1, 2).float()/255
    save_image(imgs, path, nrow=3, padding=0)
    

if __name__ == '__main__':
    keypoints0 = np.load('keypoints0.npy')
    keypoints1 = np.load('keypoints1.npy')
    matchesAB = np.load('matchesAB.npy')
    matchesBA = np.load('matchesBA.npy')
    matchProb = matchesAB * matchesBA
    a = emfactor(matchProb, f=2, ceil=0.75)
    rm0 = np.load('rm0.npy')
    rm1 = np.load('rm1.npy')
    img1 = read_image('img1.png')
    img2 = read_image('img2.png')
    
    # # raw image + all keypoints
    # plot_images([img1, img2], dpi=115)
    # plot_keypoints(keypoints0, keypoints1, pc='#00ff00')
    # # plot_matches(keypoints0, keypoints1, lc='#00ff00', lw=0.5, a=a)
    # # plot_matches(keypoints0, keypoints1, a=a)
    # # plot_matches(keypoints0, keypoints1, pc='#00ff00', lc='#00ff00', lw=0.5, a=a)
    # fig = plt.gcf()
    # fig1 = img2tensor(fig2img(fig))
    # # save_image(fig1, 'fig1.png')
    # # fig.savefig('fig.png')
    
    # raw image + all keypoints + matches
    plot_images([img1, img2], dpi=115)
    plot_keypoints(keypoints0, keypoints1, pc='#00ff00')
    plot_matches(keypoints0, keypoints1, lc='#00ff00', lw=0.5, a=a)
    fig = plt.gcf()
    # fig2 = img2tensor(fig2img(fig))
    # fig.savefig('fig.png')
    
    # response map + all keypoints
    # plot_images([rm0, rm1], cmaps='Reds')
    # fig = plt.gcf()
    # fig.savefig('rm.png')

    # figs = torch.stack((fig1, fig2), dim=0)
    # save_image(figs, 'fig.png', nrow=1, padding=0)
