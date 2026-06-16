import math

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


def init_target_grid(h, w, x_bound=(-2, 2), y_bound=(-2, 2)):
    """
    taking our target shape, it generates the complex grid, which we can apply functions on.

    If your output image width (w) is 1000 pixels wide, and your x_bounds are (-2, 2):
    Pixel 0 gets mapped to -2.0
    Pixel 500 (the exact middle) gets mapped to 0.0
    Pixel 999 (the far edge) gets mapped to 2.0
    we need boundaries because in some functions, they may go invalid(pretty big)

    h, w are the size of our actual canva, while boundaries are Domain for our math functions.
    so by increasing h, w we get a higher quality outcome, and by increasing boundaries, we can actually zoom in/out
    ai explanations in explanation file lol i'm too lazy
    """
    cx, cy = w/2, h/2

    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    def map_to_bounds(grid, c, size, bound):
        res = np.zeros_like(grid, dtype=float)
        mask_left = grid < c
        res[mask_left] = np.interp(grid[mask_left], [0, c], [bound[0], 0])
        mask_right = grid >= c
        res[mask_right] = np.interp(grid[mask_right], [c, size - 1],[0, bound[1]])
        return res

    x_scaled = map_to_bounds(j, cx, w, x_bound) # # map from 0:w to x_bound(faster than python expression) at center given
    y_scaled = map_to_bounds(i, cy, h, y_bound)
    return x_scaled+1j*y_scaled


def render_from_grid(source_image, z_source, tgt_shape, source_center=None):
    """
    expects z_src in range [-1, 1]!!!
    """
    h, w = source_image.shape[0:2]
    tgt_h, tgt_w = tgt_shape

    if source_center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = source_center
    src_half_h, src_half_w = h / 2, w / 2

    # turn the final mathematical complex coordinates back into source pixel locations
    # map from [-1, 1] to source size, centered at cx, cy
    x_src, y_src = z_source.real, z_source.imag
    x_src = (x_src * src_half_w) + cx
    y_src = (y_src * src_half_h) + cy


    # --- Catch everything out-of-bounds (including NaNs, Infs, and 1e300) ---
    # (we would've lost them anyway later in valid_mask filter, it's just that cpu can't cast them to int and would throw an exception)
    valid_mask = (x_src >= 0) & (x_src < w - 1) & (
                y_src >= 0) & (y_src < h - 1)
    x_src[~valid_mask] = 0.0
    y_src[~valid_mask] = 0.0

    # --- bilinear interpolation ---
    j0 = np.floor(x_src).astype(int)    # top line
    i0 = np.floor(y_src).astype(int)    # left line

    w_j0 = 1-(x_src-j0)     # for example if x_src=1.9, j0=1 and we want 0.1 here(far == low weight)
    w_i0 = 1-(y_src-i0)
    # w_j1 = x_src-j0         # for example if x_src=1.8, j0=1 and we want 0.8 here(close to right line=>high weight)
    # w_i1 = y_src-i0

    w_i0 = w_i0[..., np.newaxis]    # we need this extra dimension for broadcasting. (h, w, 3) * (h, w, 1) => each color is multiplied.
    # w_i1 = w_i1[..., np.newaxis]
    w_i1 = (y_src-i0)[..., np.newaxis]
    w_j0 = w_j0[..., np.newaxis]
    w_j1 = (x_src-j0)[..., np.newaxis]

    color = (
        source_image[i0, j0] * w_i0 * w_j0 +      # top left pixel
        source_image[i0, 1+j0] * w_i0 * w_j1 +      # top right pixel
        source_image[1+i0, j0] * w_i1 * w_j0 +      # bottom left pixel
        source_image[1+i0, 1+j0] * w_i1 * w_j1        # bottom right pixel
    )

    target = np.zeros((tgt_h, tgt_w, source_image.shape[2]), dtype=np.uint8)
    target[valid_mask] = color[valid_mask].astype(np.uint8)
    return target


def exp(z): return np.exp(z)
def mobius(z): return (z - 1) / (z + 1+1e-9)
def distort(z): return z + 0.1 * np.sin(z)
def log(z): return np.log(z)

def droste_map(z):
    scale_factor = 16
    log_scale = np.log(scale_factor)

    z_log = np.log(z + 1e-12)

    c = 2 * np.pi * 1j / (log_scale + 2 * np.pi * 1j)
    z_src_log = z_log / c

    real_wrapped = (z_src_log.real % log_scale) - log_scale
    imag_wrapped = z_src_log.imag

    return np.exp(real_wrapped + 1j * imag_wrapped)


def chain_transforms(*funcs):
    """allows pipeline to do multiple operations sequentially(composition): f(g(h(z)))"""
    def wrapper(z):
        for f in funcs:
            z = f(z)
        return z
    return wrapper

def cartesian_to_complex(f):
    """
    Adapter that converts a 'normal' cartesian spatial function f(x, y) -> (x_new, y_new)
    into a complex domain function that our grid engine can consume.
    """
    def wrapper(z):
        x_new, y_new = f(z.real, z.imag)
        return x_new + 1j * y_new
    return wrapper

def stretch_x(x, y):
    return 2 * x, y

complex_stretch = cartesian_to_complex(stretch_x)

def transform_image(image_, map_func, math_scale=np.pi, img_size_scale=2, source_zoom=None, center=None, x_bound=None, y_bound=None):
    if x_bound is None:
        x_bound = (-math_scale, math_scale)
    if y_bound is None:
        y_bound = (-math_scale, math_scale)

    h_tgt, w_tgt = image_.shape[0] * img_size_scale, image_.shape[1] * img_size_scale
    z = init_target_grid(h_tgt, w_tgt, x_bound=x_bound, y_bound=y_bound)

    z = map_func(z)
    if source_zoom is not None:
        z = scale_to_bounds(z, source_zoom)    # after mapping, we now scale things to defined range
    return render_from_grid(image_, z, tgt_shape=(h_tgt, w_tgt), source_center=center)


def scale_to_bounds(z, factor):
    """
    Scales the real and imaginary components independently based on their maximum
    axial limits, preventing diagonal corners from over-compressing the flat edges.
    """
    max_magnitude = np.max(np.abs(z))
    if max_magnitude == 0:
        return z

    print(z)
    # 2. Scale uniformly to preserve the angles (keeps circles round!)
    # We multiply by np.sqrt(2) so the circle expands to fill the square corners
    scaled_z = (z / max_magnitude) * factor
    return scaled_z


def escher_droste_map(R1=0.1, R2=1.0):
    """
    True Conformal Droste transform.
    Maps the annular region between R1 and R2 into an infinite spiral.
    """
    # Magnification factor between the outer and inner frame
    m = R2 / R1
    ln_m = np.log(m)

    # The magical Escher scaling factor (complex number alpha)
    # This dictates the twist angle so the edges match up perfectly
    alpha = 1 - 1j * (ln_m / (2 * np.pi))

    def wrapper(z):
        # 1. Map to Log-Polar space
        # (We add a tiny epsilon to avoid log(0) at the absolute center)
        z_log = np.log(z + 1e-12)

        # 2. Apply the Escher Twist
        z_twisted = z_log * alpha

        # 3. Seamless Modulo Wrap
        # This keeps the coordinates looping infinitely between ln(R1) and ln(R2)
        ln_R1 = np.log(R1)
        real_wrapped = ln_R1 + np.mod(z_twisted.real - ln_R1, ln_m)
        imag_wrapped = z_twisted.imag

        # 4. Project back to Cartesian space
        return np.exp(real_wrapped + 1j * imag_wrapped)

    return wrapper


if __name__=='__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(SCRIPT_DIR, "images/3b1b.png")
    source = np.array(Image.open(IMAGE_PATH).convert("RGB"))

    pipeline = chain_transforms(droste_map)
    final_image = transform_image(source, pipeline, math_scale=2*np.pi, img_size_scale=2, center=(407, 274))
    image = Image.fromarray(final_image)
    image.show()

    # steps: 1- log center=(407, 274)    2- source_center=(1075, 1875)


