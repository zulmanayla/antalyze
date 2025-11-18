# =========================
# SISIPKAN.PY — PART 1/6
# =========================

import numpy as np
import cv2
import torch
import lpips
from PIL import Image
import base64
import multiprocessing as mp

import streamlit as st

# Import worker & ACO utilities from external files
from aco_worker import aco_worker
from aco_impl import resize_for_aco
from multiprocessing import shared_memory


# Windows-safe multiprocessing
if __name__ == "__main__":
    mp.freeze_support()

# LPIPS model stored once only
if "lpips_model" not in st.session_state:
    st.session_state.lpips_model = lpips.LPIPS(net="alex")

# =========================
# SISIPKAN.PY — PART 2/6
# =========================

# Convert PIL <-> NumPy
def pil_to_np(img):
    return np.array(img)

def np_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))


# Hitung metrik (LPIPS pakai model session_state)
def evaluate_metrics(cover_np, stego_np):
    mse = np.mean((cover_np.astype("float32") - stego_np.astype("float32")) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse != 0 else float("inf")
    nc = np.sum(cover_np * stego_np) / np.sqrt(np.sum(cover_np**2) * np.sum(stego_np**2))

    lpips_fn = st.session_state.lpips_model
    img0 = torch.tensor(cover_np / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    img1 = torch.tensor(stego_np / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    lpips_score = lpips_fn(img0, img1).item()

    return {"mse": mse, "psnr": psnr, "nc": nc, "lpips": lpips_score}


# LSB — embed
def lsb_embed(cover_np, message):
    secret_bits = ''.join(format(ord(c), '08b') for c in message)
    secret_bits = [int(b) for b in secret_bits]
    length_bits = [int(b) for b in format(len(secret_bits), '016b')]
    secret_bits = length_bits + secret_bits

    stego = cover_np.copy()
    h, w, c = stego.shape
    total = h * w * c

    if len(secret_bits) > total:
        raise ValueError("Pesan terlalu besar untuk gambar")

    idx = 0
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                if idx >= len(secret_bits):
                    return stego
                stego[y, x, ch] = (stego[y, x, ch] & 0xFE) | secret_bits[idx]
                idx += 1

    return stego


# LSB — extract
def lsb_extract(stego_np):
    h, w, c = stego_np.shape
    bits = [(stego_np[y, x, ch] & 1) for y in range(h) for x in range(w) for ch in range(c)]

    length_bits = bits[:16]
    msg_len = int(''.join(map(str, length_bits)), 2)

    msg_bits = bits[16:16+msg_len]
    bytes_out = []
    for i in range(0, len(msg_bits), 8):
        chunk = msg_bits[i:i+8]
        if len(chunk) == 8:
            bytes_out.append(int(''.join(map(str, chunk)), 2))

    return bytes(bytes_out)

# =========================
# SISIPKAN.PY — PART 3/6
# =========================

st.set_page_config(page_title="Steganografi PCD", layout="wide")

# Helper load gambar background
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Path lokal Anda
image_path = "D:\\Naya Geming\\steganografi pcd\\Calming pixel art landscape.jpg"
judul_path = "D:\\Naya Geming\\steganografi pcd\\sisipkan.png"
banner_path = "D:\\Naya Geming\\steganografi pcd\\banner.png"

try:
    bg_image = get_base64_image(image_path)
    judul_path_base64 = get_base64_image(judul_path)
    banner_base64 = get_base64_image(banner_path)
except:
    bg_image = None
    judul_path_base64 = None
    banner_base64 = None

if bg_image:
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)

# Judul
if judul_path_base64:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{judul_path_base64}" width="1200">
        </div>
        """,
        unsafe_allow_html=True
    )


# ==== Global Font & Toolbar ====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quantico:wght@700&display=swap');

h3, .subheader-text, label, p {
    font-family: 'Quantico';
}

/* Toolbar Transparan */
[data-testid="stToolbar"] {
    background-color: transparent !important;
    box-shadow: none !important;
}
header[data-testid="stHeader"] {
    background-color: transparent !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


# Sidebar style (sama seperti sebelumnya)
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #87CEFA; 
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px !important;
}
h3, .subheader-text, label, p {
    color: white !important;
    font-family: 'Quantico';
}
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Steganografi (LSB & ACO)", layout="wide")


# =========================
# SISIPKAN.PY — PART 4/6
# =========================

# ==== Subheader Helpers ====
def sub_left(text):
    st.markdown(f"<h3 class='subheader-text' style='text-align:left;'>{text}</h3>", unsafe_allow_html=True)

def sub_center(text):
    st.markdown(f"<h3 class='subheader-text' style='text-align:center;'>{text}</h3>", unsafe_allow_html=True)



# Buat 2 kolom: kiri untuk upload, kanan untuk radio


# Buat 2 kolom: kiri untuk upload, kanan untuk radio
col1, col2 = st.columns([2.3, 1])  # bisa diubah proporsinya

with col1:
    st.markdown(
        "<span style='font-family:Quantico; font-size:28px;'>Upload Gambar</span>",
        unsafe_allow_html=True
    )
    uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"])

with col2:
    # CSS untuk menggeser radio button
    st.markdown(
        """
        <style>
        div[data-testid="stRadio"] {
            margin-top: 60px;   /* geser ke bawah */
            margin-left: 70px;  /* geser ke kanan */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Label kustom untuk radio button
    st.markdown(
        "<span style='font-family:Quantico; font-size:28px;'>Pilih Metode</span>",
        unsafe_allow_html=True
    )
    method = st.radio(
        "",
        ["LSB", "ACO"],
        horizontal=True
    )

# st.write("Metode dipilih:", method)
# Layout: Gambar kiri, Input pesan kanan
col_left, col_right = st.columns([1.2, 1])

with col_left:
    sub_left("<span style='font-family:Quantico;'>Gambar Asli </span>")
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")   
        st.image(image, caption="Gambar Asli", width=380)


with col_right:
    sub_left("<span style='font-family:Quantico;'>Masukkan Pesan</span>")
    message_input = st.text_area("", height=250, placeholder="Tulis pesan di sini...")

    # tombol berada di bawah input
    run_btn = st.button("🔐 Sisipkan", use_container_width=True)


# ================= CSS tombol =====================
st.markdown("""
<style>
div.stButton > button {
    background-color: rgba(255, 255, 255, 0);
    color: white;
    border: 2px solid #4CAF50;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px 20px;
    transition: 0.3s ease;
}

/* Hover → muncul warna */
div.stButton > button:hover {
    background-color: #4CAF50 !important;
    color: black !important;
}

/* Click (active) → warna hijau gelap */
div.stButton > button:active {
    background-color: #2E7D32 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)



if not uploaded_image or message_input == "":
    st.stop()

cover_np = np.array(image)

# =========================
# SISIPKAN.PY — PART 5/6
# =========================

if run_btn:

    # ---------- LSB ----------
    if method == "LSB":
        st.subheader("Hasil Steganografi (LSB)")

        stego_np = lsb_embed(cover_np, message_input)
        extracted = lsb_extract(stego_np).decode(errors="ignore")
        metrics = evaluate_metrics(cover_np, stego_np)

        st.image(stego_np, caption="Stego LSB", width=380)

        st.write(f"Pesan: {extracted}")
        st.write(f"PSNR: {metrics['psnr']:.2f} dB")
        st.write(f"MSE: {metrics['mse']:.5f}")
        st.write(f"NC: {metrics['nc']:.4f}")
        st.write(f"LPIPS: {metrics['lpips']:.4f}")

        st.stop()

    # ---------- ACO PARALLEL ----------
    st.subheader("Hasil Steganografi (ACO — Paralel)")

    parallel_ants = [10, 30, 50]
    serial_ants = [70, 100]

    # === SHARED MEMORY untuk cover_np ===
    cover_np = np.array(image)

    shm = shared_memory.SharedMemory(create=True, size=cover_np.nbytes)
    shm_arr = np.ndarray(cover_np.shape, dtype=cover_np.dtype, buffer=shm.buf)
    shm_arr[:] = cover_np[:]   # copy sekali

    # === Build task list ===
    tasks = [
        (shm.name, cover_np.shape, str(cover_np.dtype), message_input, ants)
        for ants in parallel_ants
    ]

    st.info("Memproses ACO paralel (SharedMemory) tungguinn yaa...")

    # === Parallel (3 workers) ===
    with mp.Pool(processes=3) as pool:
        parallel_results = pool.map(aco_worker, tasks)

    # === Serial untuk 70 & 100 semut ===
    serial_results = []
    for ants in serial_ants:
        r = aco_worker((shm.name, cover_np.shape, str(cover_np.dtype), message_input, ants))
        serial_results.append(r)

    # === Bersihkan shared memory ===
    shm.close()
    shm.unlink()

    # === Gabungkan semua hasil ===
    all_results = sorted(parallel_results + serial_results, key=lambda x: x["ants"])

    # === LPIPS dihitung di main process (lebih cepat) ===
    for res in all_results:

        ants = res["ants"]
        stego = res["stego_np"]
        extracted = res["extracted"]
        capacity = res["capacity"]

        # Hitung metrics lengkap pakai fungsi Streamlit-scope
        metrics = evaluate_metrics(cover_np, stego)

        st.markdown(f"<h3 style='color:white;'>ACO — {ants} Semut</h3>", unsafe_allow_html=True)
        st.image(stego, width=380, caption=f"Stego ACO ({ants} semut)")

        st.write(f"Extracted: {extracted}")
        st.write(f"Capacity: {capacity:.6f}")
        st.write(f"PSNR: {metrics['psnr']:.2f}")
        st.write(f"MSE: {metrics['mse']:.6f}")
        st.write(f"NC: {metrics['nc']:.4f}")
        st.write(f"LPIPS: {metrics['lpips']:.4f}")

        st.markdown("---")
# =========================
# SISIPKAN.PY — PART 6/6
# =========================

if __name__ == "__main__":
    mp.freeze_support()
