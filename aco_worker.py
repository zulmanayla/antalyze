# =======================================================
# aco_worker_module.py — Worker ACO super cepat (Windows-safe)
# =======================================================

import numpy as np
import cv2
from multiprocessing import shared_memory
from aco_impl import ACOSteganography, resize_for_aco


def aco_worker(args):
    """
    Worker untuk multiprocessing — tanpa Streamlit, tanpa LPIPS,
    tanpa kirim cover_np besar (pakai SharedMemory).
    """
    shm_name, shape, dtype_str, message, ants = args

    # Attach shared memory → 0 copy
    shm = shared_memory.SharedMemory(name=shm_name)
    cover = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)

    # Resize untuk ACO (lebih cepat)
    small_cover = resize_for_aco(cover, 1024)

    # Jalankan ACO
    aco = ACOSteganography(ants=ants, iterations=25)

    complex_small, _ = aco.detect_complex_regions(small_cover)

    # Scale kembali ke ukuran asli
    complex_region = cv2.resize(
        complex_small.astype(np.uint8),
        (cover.shape[1], cover.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    seed_key = f"secret_key_{ants}"

    # Embed
    stego_np, capacity = aco.embed_message(cover, message, complex_region, seed_key)

    # Extract untuk verifikasi
    extracted = aco.extract_message(stego_np, complex_region, seed_key)

    # Tutup shared memory
    shm.close()

    return {
        "ants": ants,
        "stego_np": stego_np,
        "capacity": capacity,
        "extracted": extracted
    }
