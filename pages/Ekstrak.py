import streamlit as st
from PIL import Image
import base64
from stegano import lsb
import io

# --- Convert local image to base64 ---
def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Paths for background & title ---
image_path = "Calming pixel art landscape.jpg"
judul_path = "ekstrak.png"

bg_image = get_base64_of_image(image_path)
judul_path_base64 = get_base64_of_image(judul_path)

# --- CSS: Font 1 (Quantico), Transparent Toolbar, White Subheaders ---
st.markdown("""
    <style>
    /* Import Font 1 (Quantico) */
    @import url('https://fonts.googleapis.com/css2?family=Quantico:ital,wght@0,400;0,700;1,400;1,700&display=swap');

    /* ===== Font 1 untuk semua Subheader ===== */
    h3 {
        font-family: 'Quantico', sans-serif !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        text-align: left;
    }

    /* ===== Toolbar Transparan ===== */
    [data-testid="stToolbar"] {
        background-color: transparent !important;
        backdrop-filter: blur(0px) !important;
        box-shadow: none !important;
    }

    /* Hilangkan garis bawah header */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Page background CSS ---
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Sidebar CSS ---
st.markdown("""
<style>
/* ===== Preload Material Icons agar tidak glitch ===== */
@font-face {
    font-family: 'Material Icons';
    font-style: normal;
    font-weight: 400;
    src: url(https://fonts.gstatic.com/s/materialicons/v126/flUhRq6tzZclQEJ-Vdg-IuiaDsNcIhQ8tQ.woff2) format('woff2');
}
.material-icons {
    font-family: 'Material Icons' !important;
    font-weight: normal;
    font-style: normal;
    font-size: 24px;
    display: inline-block;
    line-height: 1;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
}

/* ===== Sidebar Styling ===== */
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

div[data-testid="stRadio"] input:checked + div {
    color: #FFFFFF !important;
    font-weight: bold;
}

/* ===== Toolbar Transparan ===== */
[data-testid="stToolbar"] {
    background-color: transparent !important;
    backdrop-filter: blur(0px) !important;
    box-shadow: none !important;
}

/* Hilangkan garis bawah header */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

/* ===== Perbaikan Ikon Collapse (selalu pakai Material Icons) ===== */
button[kind="headerNoPadding"] span[data-testid="stIconMaterial"] {
    font-family: 'Material Icons' !important;
    font-size: 24px !important;
    color: white !important;
    opacity: 0.9;
    margin-top: 2px;
}
</style>

<!-- Force preload font lebih awal -->
<link rel="preload" href="https://fonts.gstatic.com/s/materialicons/v126/flUhRq6tzZclQEJ-Vdg-IuiaDsNcIhQ8tQ.woff2" as="font" type="font/woff2" crossorigin>
""", unsafe_allow_html=True)

# --- Title image ---
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{judul_path_base64}" width="1800">
    </div>
    """,
    unsafe_allow_html=True
)

# --- Upload section ---
st.subheader("Upload Gambar")
uploaded_image = st.file_uploader(
    "Upload gambar yang mau kamu ekstrak pesannya.", 
    type=["jpg", "jpeg", "png"]
)

# --- If image uploaded, show two columns ---
if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    # --- LEFT COLUMN: Original image with centered subheader ---
    with col1:
        st.markdown(
            """
            <h3 style='text-align: center; color: white; font-family: Quantico; font-weight: 700;'>
                Gambar Asli
            </h3>
            """,
            unsafe_allow_html=True
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        st.markdown(
            f"""
            <div style="padding: 10px;">
                <img src="data:image/png;base64,{img_b64}" 
                    style="width:100%; border-radius:10px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- RIGHT COLUMN: Extract hidden message ---
    with col2:
    # Geser container kanan sedikit ke bawah
        st.markdown("""
        <h3 style='text-align: center; color: white; font-family: Quantico; font-weight: 700; margin-bottom: 10px;'>
            Hasil Ekstraksi Teks
        </h3>
    """, unsafe_allow_html=True)


        with st.spinner("Mengekstraksi pesan tersembunyi..."):
            try:
                pesan_tersembunyi = lsb.reveal(uploaded_image)
                if pesan_tersembunyi and pesan_tersembunyi.strip():
                    st.success("Pesan tersembunyi terdeteksi:")
                    st.write(pesan_tersembunyi)
                else:
                    st.warning("Tidak ditemukan pesan tersembunyi.")
            except Exception:
                st.error("Tidak ditemukan pesan tersembunyi.")




