from dataclasses import dataclass
from typing import Callable
import numpy as np
from src.app.convolution import convolve, convolve_separable

MASK_SIZES = [3, 5, 7, 9]


@dataclass(frozen=True)
class FilterParams:
    mask_size: int
    sigma: float
    fast: bool


@dataclass(frozen=True)
class FilterSpec:
    name: str
    func: Callable[[np.ndarray, FilterParams], np.ndarray]
    uses_mask: bool = True
    uses_sigma: bool = False
    uses_fast: bool = True


# dodatkowe maski
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
LAPLACE = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])


def average_kernel(n):
    return np.ones((n, n)) / n ** 2


def gaussian_kernel(n, sigma):
    ax = np.arange(n) - n // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_kernel_1d(sigma):
    half = int(np.ceil(3.0 * sigma))
    x = np.arange(-half, half + 1)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def sharpen_kernel(n):
    k = -np.ones((n, n))
    k[n // 2, n // 2] = n ** 2   # suma maski = 1
    return k


def average(img, p):
    return convolve(img, average_kernel(p.mask_size), fast=p.fast)


def gaussian(img, p):
    return convolve(img, gaussian_kernel(p.mask_size, p.sigma), fast=p.fast)


def gaussian_1d(img, p):
    return convolve_separable(img, gaussian_kernel_1d(p.sigma))


def sharpen(img, p):
    return convolve(img, sharpen_kernel(p.mask_size), fast=p.fast)


def sobel(img, p):
    gx = convolve(img, SOBEL_X, raw_output=True, fast=p.fast)
    gy = convolve(img, SOBEL_Y, raw_output=True, fast=p.fast)
    return np.clip(np.sqrt(gx ** 2 + gy ** 2), 0, 255).astype(np.uint8)


def laplace(img, p):
    return convolve(img, LAPLACE, fast=p.fast)

_SPECS = [
    FilterSpec("Uśredniający", average),
    FilterSpec("Gaussa", gaussian, uses_sigma=True),
    FilterSpec("Gaussa1d", gaussian_1d, uses_mask=False, uses_sigma=True, uses_fast=False),
    FilterSpec("Wyostrzający", sharpen),
    FilterSpec("Sobela", sobel, uses_mask=False),      # maska zawsze 3x3
    FilterSpec("Laplace", laplace, uses_mask=False),   # maska zawsze 3x3
]

FILTERS: dict[str, FilterSpec] = {s.name: s for s in _SPECS}
FILTER_NAMES = list(FILTERS)