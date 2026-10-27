import numpy as np

# GENAI Podpowiedziało zamysł i idee zoptymalizowanego algorytmu splotu, gdzie nie przechodzimy po pixelach,
# ale po masce mnożąc jej wartości przez cały obraz ciągle dodając do finalnej wersji.

def convolve(image, kernel, raw_output=False, fast=True):
    k_h, k_w = kernel.shape
    pad = k_h // 2
    height, width = image.shape[:2]
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

    if fast:
        # przechodzimy po masce, a nie po pikselach obrazu
        output = np.zeros(image.shape, dtype=np.float32)
        for i in range(k_h):
            for j in range(k_w):
                # pixel kernela * każdy pixel obrazka od pixela i,j
                output += kernel[i, j] * padded[i:i + height, j:j + width, :]
    else:
        # wolna wersja: pętla po każdym pikselu
        output = np.zeros(image.shape, dtype=np.float64)
        for y in range(height):
            for x in range(width):
                pixel_sum = 0.0
                for ky in range(k_h):
                    for kx in range(k_w):
                        pixel_sum += kernel[ky, kx] * padded[y + ky, x + kx]
                output[y, x] = pixel_sum

    return output if raw_output else np.clip(output, 0, 255).astype(np.uint8)


def convolve_separable(image, kernel_1d):
    size = kernel_1d.shape[0]
    pad = size // 2
    height, width = image.shape[:2]

    padded_v = np.pad(image, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    tmp = np.zeros(image.shape, dtype=np.float32)
    for i in range(size):
        tmp += kernel_1d[i] * padded_v[i:i + height, :, :]

    padded_h = np.pad(tmp, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros(image.shape, dtype=np.float32)
    for j in range(size):
        out += kernel_1d[j] * padded_h[:, j:j + width, :]

    return np.clip(out, 0, 255).astype(np.uint8)