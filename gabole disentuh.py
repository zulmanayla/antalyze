import streamlit as st
from PIL import Image
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_score, mean_squared_error as mse_score
import base64
import io
import textwrap


# ==== Base64 Loader ====
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded


# ==== Background & Header ====
image_path = "D:\\Naya Geming\\steganografi pcd\\Calming pixel art landscape.jpg"
judul_path = "D:\\Naya Geming\\steganografi pcd\\sisipkan.png"

bg_image = get_base64_of_image(image_path)
judul_path_base64 = get_base64_of_image(judul_path)

# Background
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

# Header Image
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{judul_path_base64}" width="1800">
    </div>
    """,
    unsafe_allow_html=True
)


# ==== Sidebar ====
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #87CEFA; 
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-family: 'Segoe UI', sans-serif;
    font-size: 25px !important;
}
div[data-testid="stRadio"] label:hover {
    background-color: #1E90FF;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==== Global Font & Toolbar ====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quantico:wght@700&display=swap');

h3, .subheader-text, label, p {
    color: white !important;
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


# ==== Subheader Helpers ====
def sub_left(text):
    st.markdown(f"<h3 class='subheader-text' style='text-align:left;'>{text}</h3>", unsafe_allow_html=True)

def sub_center(text):
    st.markdown(f"<h3 class='subheader-text' style='text-align:center;'>{text}</h3>", unsafe_allow_html=True)



# ============================================================
# ========== LSB FUNCTIONS ===================================
# ============================================================

def message_to_binary(message):
    return ''.join([format(ord(i), '08b') for i in message])


def encode_image(image, messages):
    np_image = np.array(image.convert("RGB"))
    combined_message = '|'.join(messages) + "#####"
    binary_message = message_to_binary(combined_message)

    idx = 0
    total = len(binary_message)

    for i in range(np_image.shape[0]):
        for j in range(np_image.shape[1]):
            for k in range(3):
                if idx < total:
                    pixel = np_image[i, j, k]
                    pixel_bin = format(pixel, '08b')
                    new_bin = pixel_bin[:-1] + binary_message[idx]
                    np_image[i, j, k] = int(new_bin, 2)
                    idx += 1
        if idx >= total:
            break

    return Image.fromarray(np_image)


def binary_to_string(binary_data):
    chars = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
    decoded = ''.join([chr(int(c, 2)) for c in chars])
    return decoded.split("#####")[0]


def decode_and_evaluate(original_image, encoded_image):
    np_encoded = np.array(encoded_image)
    binary_data = ""

    for i in range(np_encoded.shape[0]):
        for j in range(np_encoded.shape[1]):
            for k in range(3):
                pixel_bin = format(np_encoded[i, j, k], '08b')
                binary_data += pixel_bin[-1]

    hidden_msg = binary_to_string(binary_data)

    ori = np.array(original_image)
    enc = np.array(encoded_image)

    mse = mse_score(ori, enc)
    psnr_val = psnr_score(ori, enc)

    bits = len(binary_data)
    h, w, c = ori.shape
    capacity = bits / (h * w * c)
    bpp = bits / (h * w)

    loss_fn = lpips.LPIPS(net='alex')
    o_tensor = torch.tensor(ori.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    e_tensor = torch.tensor(enc.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    lpips_val = loss_fn(o_tensor, e_tensor).item()

    nc = np.corrcoef(ori.flatten(), enc.flatten())[0, 1]

    return {
        "hidden_message": hidden_msg,
        "capacity": capacity,
        "mse": mse,
        "psnr": psnr_val,
        "nc": nc,
        "lpips": lpips_val,
        "bpp": bpp
    }



# ============================================================
# =========================== UI =============================
# ============================================================

st.set_page_config(page_title="LSB Steganografi", layout="wide")

# Upload Image
sub_left("<span style='font-family:Quantico;'>Upload Gambar</span>")
uploaded_image = st.file_uploader("", type=["png", "jpg", "jpeg"])

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

# ============================================================
# ========================== PROCESS =========================
# ============================================================

if run_btn:

    if not uploaded_image:
        st.error("Silakan upload gambar terlebih dahulu.")
    elif not message_input.strip():
        st.error("Pesan tidak boleh kosong.")
    else:
        pesan_list = [p.strip() for p in message_input.split(",")]
        encoded_img = encode_image(image, pesan_list)
        results = decode_and_evaluate(image, encoded_img)

        st.markdown("---")
        st.markdown(
                f"""
                <h2 style='text-align:center; color:#B6DF29; font-family:Quantico; font-weight:700;'>
                Steganografi dengan Least Significant Bit (LSB)

                </h2>

                """,
                unsafe_allow_html=True
            )
        # st.subheader("Steganografi dengan Least Significant Bit (LSB) ")
        

        col1, col2 = st.columns(2)

        # --- LEFT ORIGINAL ---
        with col1:
            buf_ori = io.BytesIO()
            image.save(buf_ori, format="PNG")
            ori_base64 = base64.b64encode(buf_ori.getvalue()).decode()

            st.markdown(
                f"""
                <h3 style='text-align:center; color:#B6DF29; font-family:Quantico; font-weight:700;'>
                    Gambar Asli
                </h3>

                <div style="display:flex; justify-content:center; align-items:center;">
                    <img src="data:image/png;base64,{ori_base64}"
                        style="
                            max-width:350px;
                            height:auto;
                            margin-top:20px;
                            border-radius:10px;
                            box-shadow:0 4px 10px rgba(0,0,0,0.3);
                        " />
                </div>
                """,
                unsafe_allow_html=True
            )

        # --- RIGHT ENCODED ---
        with col2:
            buf = io.BytesIO()
            encoded_img.save(buf, format="PNG")
            encoded_base64 = base64.b64encode(buf.getvalue()).decode()

            st.markdown(
                f"""
                <h3 style='text-align:center; color:#B6DF29; font-family:Quantico; font-weight:700;'>
                    Gambar Setelah Penyisipan
                </h3>

                <div style="display:flex; justify-content:center; align-items:center;">
                    <img src="data:image/png;base64,{encoded_base64}"
                        style="
                            max-width:350px;
                            height:auto;
                            margin-top:20px;
                            border-radius:10px;
                            box-shadow:0 4px 10px rgba(0,0,0,0.3);
                        " />
                </div>
                """,
                unsafe_allow_html=True
            )

        # ============================
        #   BANNER HASIL EVALUASI
        # ============================

        with open("D:\\Naya Geming\\steganografi pcd\\banner.png", "rb") as f:
            banner_base64 = base64.b64encode(f.read()).decode()

        html_block = """
        <div style="
            background-image: url('data:image/png;base64,{banner}');
            background-size: cover;
            padding: 40px;
            border-radius: 15px;
            margin-top: 20px;
        ">
        
        <!-- INNER WRAPPER untuk memotong kanan kiri -->
        <div style="
            padding-left: 30px;
            padding-right: 30px;
        ">
            <h2 style="
                text-align:center; 
                font-family: 'Quantico'; 
                color:#B6DF29;
                margin-top: 35px;  
                margin-bottom: 45px;
            ">
                📊 Hasil Evaluasi LSB
            </h2>

            <div style="
                font-family: 'Lato'; 
                font-size: 20px; 
                color: white; 
                line-height: 1.8;
                text-align:center;
            ">
                <p><b>Pesan tersembunyi:</b> {msg}</p>
                <p><b>MSE:</b> {mse:.4f}</p>
                <p><b>PSNR:</b> {psnr:.2f} dB</p>
                <p><b>NC:</b> {nc:.4f}</p>
                <p><b>LPIPS:</b> {lpips:.4f}</p>
                <p><b>Capacity:</b> {cap:.6f}</p>
                <p><b>BPP:</b> {bpp:.6f}</p>
            </div>
        </div>
        """.format(
            banner=banner_base64,
            msg=results["hidden_message"],
            mse=results["mse"],
            psnr=results["psnr"],
            nc=results["nc"],
            lpips=results["lpips"],
            cap=results["capacity"],
            bpp=results["bpp"]
        )
    
    st.markdown(html_block, unsafe_allow_html=True)
