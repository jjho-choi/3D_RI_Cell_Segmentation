import numpy as np
import h5py


def _padding(img, tiles):
    d, h, w = img.shape
    h_size = int(round(h / tiles[0]))
    w_size = int(round(w / tiles[1]))
    tile_len = max(h_size, w_size)
    h_size = tile_len * tiles[0]
    w_size = tile_len * tiles[1]

    if h_size > h:
        pad_size = h_size - h 
        img = np.pad(img, ((0, 0), (0, pad_size), (0, 0)), 'constant')
    if w_size > w:
        pad_size = w_size - w
        img = np.pad(img, ((0, 0), (0, 0), (0, pad_size)), 'constant')

    return img


def temp(x):
    return x


def _sym_padding(patch_size, img):
    pad_size = patch_size // 2
    return np.pad(img, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'symmetric')


def _patch_offset_generation(patch_size, img):
    z, y, x = img.shape
    x_n = y_n = patch_size // 2

    patch_offset = []
    for h in range((y - patch_size) // y_n + 2):
        for w in range((x - patch_size) // x_n + 2):
            y_offset = y_n * h
            x_offset = x_n * w

            if y_offset + patch_size > y:
                continue
            if x_offset + patch_size > x:
                continue
            patch_offset.append((y_offset, x_offset))

    return patch_offset


def get_time(file):
    with h5py.File(file, 'r') as img:
        times = list(img['Data/3D'])
    return times
