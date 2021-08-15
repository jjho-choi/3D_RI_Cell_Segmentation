
import random

from scipy.ndimage import zoom


class Crop:
    """Cropping on numpy image

    Args :
        crop_shape (tuple, list) : Crop shape (e.g. 2D -> (H, W), 3d -> (Z, H, W))
        center (tuple of bool) : Flag for center/random crop
            Each tuple must be (D, H, W) format
            If each value is True, then that dimension crop centered.

    Example::
        >>> img = get_2d_npy("*.npy")
        >>> img.shape
        >>> [3, 224, 224]
        >>> crop = Crop([64, 64])
        >>> crop_img = crop(img)
        >>> crop_img.shape
        >>> [3, 64, 64]
        >>> img = get_3d_npy("*.npy")
        >>> img.shape
        >>> [1, 64, 128, 128]
        >>> crop = Crop([32, 64, 64])
        >>> crop(img).shape
        >>> [1, 32, 64, 64]
    """
    def __init__(self, crop_shape, center=(True, False, False), **kwargs):
        assert len(center) == 3

        self.crop_shape = crop_shape
        self.center = center
        self.ndim = len(crop_shape) + 1

        if self.ndim == 3:
            self._crop_fn = self._crop_2d
        elif self.ndim == 4:
            self._crop_fn = self._crop_3d
        else:
            raise ValueError("Crop shape must be 2d or 3d")

    def _get_range(self, center, shape, crop_shape):
        if center:
            middle, half = shape // 2, crop_shape // 2
            start = middle - half
            end = middle + half
        else:
            start = random.randint(0, shape - crop_shape)
            end = start + crop_shape
        return start, end

    def _crop_2d(self, img):
        c, x, y = img.shape
        sx, ex = self._get_range(self.center[1], x, self.crop_shape[0])
        sy, ey = self._get_range(self.center[2], y, self.crop_shape[1])
        return img[:, sx:ex, sy:ey]

    def _crop_3d(self, img):
        c, z, x, y = img.shape
        sz, ez = self._get_range(self.center[0], z, self.crop_shape[0])
        sx, ex = self._get_range(self.center[1], x, self.crop_shape[1])
        sy, ey = self._get_range(self.center[2], y, self.crop_shape[2])
        return img[:, sz:ez, sx:ex, sy:ey]

    def __call__(self, img):
        """
        Args :
            img (np.array) : Input image (e.g, 2D -> (C, H, W), 3d -> (C, D, H, W))

        Returns :
            :class: `np.ndarray`:
                Cropped image

        """
        assert self.ndim == img.ndim
        return self._crop_fn(img)


class Resize:
    """ resize a numpy image

    Args :
        crop_shape (tuple, list) : Crop shape (e.g. 2D -> (H, W), 3d -> (Z, H, W))

    Example::
        >>> img.shape
        >>> [1, 40, 80, 80]
        >>> resize = Resize(30, 40, 40)
        >>> resized_img = resize(img)
        >>> resized_img.shape
        >>> [1, 30, 40, 40]
    """

    def __init__(self, zoom_shape, mode='nearest', order=1, **kwargs):
        self.zoom_shape = zoom_shape
        self.ndim = len(zoom_shape) + 1
        self.mode = mode
        self.order = order

        if self.ndim == 3:
            self._resize_fn = self._resize_2d
        elif self.ndim == 4:
            self._resize_fn = self._resize_3d
        else:
            raise ValueError("Resize shape must be 2d or 3d")

        if self.mode not in ['reflect', 'constant', 'nearest', 'mirror', 'wrap']:
            raise ValueError("mode must be one of reflect, constant, nearest, mirror, or wrap")

        if self.order < 0 or self.order > 6:
            raise ValueError("order mus be in the range 0-5")

    def _resize_2d(self, img):
        c, x, y = img.shape
        return zoom(img, (1, self.zoom_shape[0] / x, self.zoom_shape[1] / y), mode=self.mode, order=self.order)

    def _resize_3d(self, img):
        c, z, x, y = img.shape
        return zoom(img, (1, self.zoom_shape[0] / z, self.zoom_shape[1] / x, self.zoom_shape[2] / y),
                    mode=self.mode, order=self.order)

    def __call__(self, img):
        """
        Args :
            img (np.array) : Input image (e.g, 2D -> (C, H, W), 3d -> (C, D, H, W))

        Returns :
            :class: `np.ndarray`:
                Resized image

        """
        assert self.ndim == img.ndim
        return self._resize_fn(img)
