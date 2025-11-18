# aco_impl.py  — NUMBA-ACCELERATED VERSION
# Compatible with existing aco_worker and shared-memory approach.
# Requires: numba

import numpy as np
import cv2
from numba import njit, prange, int64, float64

# ------------------------------
# Low-level JIT helpers
# ------------------------------

@njit(parallel=True, fastmath=True)
def sobel_magnitude_jit(gray):
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float64)

    for y in prange(1, h-1):
        for x in range(1, w-1):
            gx = (
                -1.0 * gray[y-1, x-1] + 1.0 * gray[y-1, x+1]
                -2.0 * gray[y,   x-1] + 2.0 * gray[y,   x+1]
                -1.0 * gray[y+1, x-1] + 1.0 * gray[y+1, x+1]
            )
            gy = (
                -1.0 * gray[y-1, x-1] -2.0 * gray[y-1, x] -1.0 * gray[y-1, x+1]
                +1.0 * gray[y+1, x-1] +2.0 * gray[y+1, x] +1.0 * gray[y+1, x+1]
            )
            out[y, x] = np.sqrt(gx * gx + gy * gy)
    return out

@njit(fastmath=True)
def normalize_array_jit(arr):
    amin = arr.min()
    amax = arr.max()
    if amax - amin <= 1e-10:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin + 1e-12)

@njit
def init_pheromone_jit(shape0, shape1, init_val=0.1):
    pher = np.empty((shape0, shape1), dtype=np.float64)
    for i in range(shape0):
        for j in range(shape1):
            pher[i, j] = init_val
    return pher

@njit
def get_neighbors_positions(y, x, h, w):
    # return a small fixed-size list with -1 as sentinel if short
    # We'll return arrays of size 8 with sentinel (-1,-1) for invalid
    ny = np.empty(8, dtype=np.int64)
    nx = np.empty(8, dtype=np.int64)
    idx = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy = y + dy
            xx = x + dx
            if 0 <= yy < h and 0 <= xx < w:
                ny[idx] = yy
                nx[idx] = xx
            else:
                ny[idx] = -1
                nx[idx] = -1
            idx += 1
    return ny, nx

@njit(fastmath=True)
def choose_next_position(y, x, pher, heuristic, alpha, beta):
    h, w = pher.shape
    ny, nx = get_neighbors_positions(y, x, h, w)
    # compute unnormalized probability
    probs = np.empty(8, dtype=np.float64)
    total = 0.0
    for i in range(8):
        if ny[i] == -1:
            probs[i] = 0.0
        else:
            t = pher[ny[i], nx[i]]
            e = heuristic[ny[i], nx[i]]
            val = (t ** alpha) * (e ** beta)
            probs[i] = val
            total += val
    if total <= 0.0:
        # fallback: choose first valid neighbor uniformly
        cnt = 0
        for i in range(8):
            if ny[i] != -1:
                cnt += 1
        if cnt == 0:
            return y, x
        pick = np.random.randint(0, cnt)
        c = 0
        for i in range(8):
            if ny[i] != -1:
                if c == pick:
                    return ny[i], nx[i]
                c += 1
    # normalize and choose by cumulative
    cum = 0.0
    r = np.random.random()
    for i in range(8):
        if probs[i] <= 0:
            continue
        p = probs[i] / total
        cum += p
        if r <= cum:
            return ny[i], nx[i]
    # fallback to last valid
    for i in range(7, -1, -1):
        if ny[i] != -1:
            return ny[i], nx[i]
    return y, x

@njit
def accumulate_delta(delta, path_y, path_x, heuristic):
    # path_y/x are arrays of length L (with real positions)
    L = path_y.shape[0]
    for i in range(L):
        y = path_y[i]
        x = path_x[i]
        delta[y, x] += heuristic[y, x]

@njit
def update_pheromone_jit(pheromone, delta, rho, psi, init_val):
    # evaporate
    h, w = pheromone.shape
    for i in range(h):
        for j in range(w):
            pheromone[i, j] = pheromone[i, j] * (1.0 - rho)
    # add delta
    for i in range(h):
        for j in range(w):
            pheromone[i, j] += delta[i, j]
    # mixing with initialization (psi)
    for i in range(h):
        for j in range(w):
            pheromone[i, j] = pheromone[i, j] * (1.0 - psi) + psi * init_val

# ------------------------------
# High-level JIT ACO routines
# ------------------------------

@njit
def single_iteration_ants(pheromone, heuristic, ants, steps_per_ant, alpha, beta):
    """
    Simulate ants for a single iteration and produce delta pheromone.
    Returns delta (same shape as pheromone) and ant_journeys_count (not used heavy).
    """
    h, w = pheromone.shape
    delta = np.zeros_like(pheromone)
    # We'll store per-ant small paths in temporary arrays (max steps)
    for a in range(ants):
        # random start
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        # preallocate path
        py = np.empty(steps_per_ant, dtype=np.int64)
        px = np.empty(steps_per_ant, dtype=np.int64)
        step = 0
        py[step] = y
        px[step] = x
        step += 1
        for s in range(1, steps_per_ant):
            ny, nx = choose_next_position(y, x, pheromone, heuristic, alpha, beta)
            y, x = ny, nx
            py[step] = y
            px[step] = x
            step += 1
        # accumulate into delta: only first 'step' entries valid
        # create views of proper length
        py_view = py[:step].copy()
        px_view = px[:step].copy()
        accumulate_delta(delta, py_view, px_view, heuristic)
    return delta

# ------------------------------
# Class ACOSteganography (Numba-accelerated internals)
# ------------------------------

class ACOSteganography:
    def __init__(self, alpha=1.0, beta=1.0, rho=0.1, psi=0.05, ants=50, iterations=25):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.psi = float(psi)
        self.ants = int(ants)
        self.iterations = int(iterations)
        self.ant_journeys = []

    def calculate_heuristic(self, image):
        # image may be RGB or grayscale (np.uint8)
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        mag = sobel_magnitude_jit(gray)
        norm = normalize_array_jit(mag)
        return norm

    def initialize_pheromone(self, shape):
        return init_pheromone_jit(shape[0], shape[1], init_val=0.1)

    def detect_complex_regions(self, image):
        # image = small resized image
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = image.astype(np.float64)

        heuristic = sobel_magnitude_jit(gray)
        heuristic = normalize_array_jit(heuristic)

        pheromone = init_pheromone_jit(gray.shape[0], gray.shape[1], init_val=0.1)

        # iterate ACO using JITed single_iteration_ants
        for it in range(self.iterations):
            delta = single_iteration_ants(
                pheromone, heuristic, self.ants, 30, self.alpha, self.beta
            )
            update_pheromone_jit(pheromone, delta, self.rho, self.psi, 0.1)

        # threshold to complex region
        thresh = np.mean(pheromone) * 1.2
        complex_region = (pheromone >= thresh).astype(np.uint8)
        return complex_region, pheromone

    def embed_message(self, cover_image, secret_message, complex_region=None, seed_key=None):
        # same logic as original (pure numpy); keep readable
        if complex_region is None:
            complex_region, _ = self.detect_complex_regions(cover_image)

        if isinstance(secret_message, str):
            secret_bits = ''.join(format(ord(c), '08b') for c in secret_message)
        else:
            secret_bits = ''.join(format(b, '08b') for b in secret_message)
        secret_bits = [int(b) for b in secret_bits]

        length_bits = [int(b) for b in format(len(secret_bits), '016b')]
        secret_bits = length_bits + secret_bits

        stego_image = cover_image.copy().astype(np.int32)
        if stego_image.ndim == 2:
            stego_image = np.expand_dims(stego_image, axis=-1)

        h, w, channels = stego_image.shape
        secret_idx = 0

        # positions using numpy for speed
        ys, xs = np.where(complex_region == 1)
        complex_positions = list(zip(ys.tolist(), xs.tolist()))
        ys2, xs2 = np.where(complex_region == 0)
        non_complex_positions = list(zip(ys2.tolist(), xs2.tolist()))

        if seed_key is not None:
            np.random.seed(hash(seed_key) % (2**32))
        np.random.shuffle(complex_positions)
        np.random.shuffle(non_complex_positions)

        positions = complex_positions + non_complex_positions

        for pos in positions:
            y, x = pos
            for c in range(channels):
                if secret_idx >= len(secret_bits):
                    break
                pixel_val = int(stego_image[y, x, c])
                bit = secret_bits[secret_idx]
                stego_image[y, x, c] = (pixel_val & 0xFE) | bit
                secret_idx += 1
            if secret_idx >= len(secret_bits):
                break

        capacity = secret_idx / (h * w * channels)
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        return stego_image.squeeze(), capacity

    def extract_message(self, stego_image, complex_region, seed_key=None):
        if stego_image.ndim == 2:
            stego_image = np.expand_dims(stego_image, axis=-1)

        height, width, channels = stego_image.shape
        ys, xs = np.where(complex_region == 1)
        complex_positions = list(zip(ys.tolist(), xs.tolist()))
        ys2, xs2 = np.where(complex_region == 0)
        non_complex_positions = list(zip(ys2.tolist(), xs2.tolist()))

        if seed_key is not None:
            np.random.seed(hash(seed_key) % (2**32))
        np.random.shuffle(complex_positions)
        np.random.shuffle(non_complex_positions)

        positions = complex_positions + non_complex_positions

        extracted_bits = []
        limit = 16 + 100000
        for pos in positions:
            y, x = pos
            for c in range(channels):
                pixel_val = stego_image[y, x, c]
                extracted_bits.append(int(pixel_val & 1))
                if len(extracted_bits) >= limit:
                    break
            if len(extracted_bits) >= limit:
                break

        if len(extracted_bits) < 16:
            return b''

        length_bits = extracted_bits[:16]
        message_length = int(''.join(map(str, length_bits)), 2)

        if len(extracted_bits) < 16 + message_length:
            return b''

        message_bits = extracted_bits[16:16 + message_length]
        message_bytes = []
        for i in range(0, len(message_bits), 8):
            byte_bits = message_bits[i:i+8]
            if len(byte_bits) == 8:
                byte = int(''.join(map(str, byte_bits)), 2)
                message_bytes.append(byte)

        return bytes(message_bytes)


def resize_for_aco(img, max_dim=1024):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
