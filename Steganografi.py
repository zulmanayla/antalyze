import streamlit as st
import base64

st.set_page_config(
    page_title="Steganografi PCD",
    page_icon="🔐",
    layout="wide",
)

# Membaca gambar lokal dan ubah jadi base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Gambar lokal
image_path = "antalyze/Calming pixel art landscape.jpg"
judul_path = "antalyze/judul.png"
teks_path = "antalyze/Pink and Blue Colorful Playful Cute Pixel Illustrative Trivia Quiz Presentation.png"
wm_path = "antalyze/wm.png"

bg_image = get_base64_of_image(image_path)
judul_base64 = get_base64_of_image(judul_path)
teks_path_base64 = get_base64_of_image(teks_path)
wm_path_base64 = get_base64_of_image(wm_path)

# CSS background halaman utama
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

# ===== Styling Sidebar + Toolbar (dengan preload font Material Icons) =====
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
button[kind="header"] div[data-testid="stIconMaterial"] {
    font-family: 'Material Icons' !important;
    font-size: 24px !important;
    color: white !important;
    opacity: 0.9;
    margin-top: 2px;
}

/* Force font tetap aktif meski Streamlit rerun halaman */
span[data-testid="stIconMaterial"]::before {
    font-family: 'Material Icons' !important;
    content: attr(aria-label);
}
</style>

<!-- Force preload font lebih awal -->
<link rel="preload" href="https://fonts.gstatic.com/s/materialicons/v126/flUhRq6tzZclQEJ-Vdg-IuiaDsNcIhQ8tQ.woff2" as="font" type="font/woff2" crossorigin>
""", unsafe_allow_html=True)

# ===== Konten utama =====
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{judul_base64}" width="1300">
        <br><br> 
        <img src="data:image/png;base64,{teks_path_base64}" width="1100">
        <br><br><br><br><br><br><br><br><br><br>
        <img src="data:image/png;base64,{wm_path_base64}" width="2100">
    </div>
    """,
    unsafe_allow_html=True
)

