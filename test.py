import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hệ Thống Định Giá Xe Máy",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS giao diện chuyên nghiệp
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .result-card {
        background-color: #f8f9fa;
        border-left: 6px solid #2a5298;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .price-tag {
        color: #e63946;
        font-size: 2.5rem;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. TẢI MÔ HÌNH DỰ ĐOÁN
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "xgb_price_model.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy file mô hình tại: `{model_path}`")
        return None

    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # Trường hợp file lưu dạng dictionary
        if isinstance(data, dict):
            for key in ["model", "estimator", "xgb_model", "xgb"]:
                if key in data:
                    return data[key]
            st.error(
                f"⚠️ File .pkl là Dictionary nhưng không chứa key model quen thuộc. Các key: {list(data.keys())}"
            )
            return None

        return data
    except Exception as e:
        st.error(f"❌ Lỗi khi nạp mô hình: {e}")
        return None


model = load_model()

# ---------------------------------------------------------
# 3. CÁC TÍNH NĂNG MÔ HÌNH (81 Features - Chuẩn UTF-8)
# ---------------------------------------------------------
FEATURE_NAMES = [
    "Tuoi_xe_final",
    "Km_final",
    "Dung_tich_ord",
    "Dung_tich_khong_ro",
    "Phan_khuc_bin",
    "Nam_khong_ro",
    "Km_lam_tron",
    "Loại xe_Tay ga",
    "Loại xe_Xe số",
    "Xuat_xu_clean_Không rõ",
    "Xuat_xu_clean_Mỹ",
    "Xuat_xu_clean_Nhật Bản",
    "Xuat_xu_clean_Nước khác",
    "Xuat_xu_clean_Thái Lan",
    "Xuat_xu_clean_Trung Quốc",
    "Xuat_xu_clean_Việt Nam",
    "Xuat_xu_clean_Đài Loan",
    "Xuat_xu_clean_Đức",
    "Xuat_xu_clean_Ấn Độ",
    "Hang_group_Ducati",
    "Hang_group_Honda",
    "Hang_group_Kawasaki",
    "Hang_group_Khác",
    "Hang_group_Kymco",
    "Hang_group_Piaggio",
    "Hang_group_SYM",
    "Hang_group_Suzuki",
    "Hang_group_Yamaha",
    "Dong_group_@",
    "Dong_group_Air Blade",
    "Dong_group_Attila",
    "Dong_group_Blade",
    "Dong_group_CB",
    "Dong_group_CBR",
    "Dong_group_Chaly",
    "Dong_group_Click",
    "Dong_group_Cub",
    "Dong_group_Dream",
    "Dong_group_Dylan",
    "Dong_group_Elegant",
    "Dong_group_Elizabeth",
    "Dong_group_Exciter",
    "Dong_group_FZ",
    "Dong_group_Future",
    "Dong_group_GSX",
    "Dong_group_GTS",
    "Dong_group_Grande",
    "Dong_group_Hayate",
    "Dong_group_Janus",
    "Dong_group_Jupiter",
    "Dong_group_Khác",
    "Dong_group_LX",
    "Dong_group_Lead",
    "Dong_group_Liberty",
    "Dong_group_Luvias",
    "Dong_group_MSX 125",
    "Dong_group_Mio",
    "Dong_group_Nouvo",
    "Dong_group_Nozza",
    "Dong_group_Nvx",
    "Dong_group_PCX",
    "Dong_group_PG-1",
    "Dong_group_PS",
    "Dong_group_R",
    "Dong_group_Raider",
    "Dong_group_SH",
    "Dong_group_SH Mode",
    "Dong_group_Satria",
    "Dong_group_Shark",
    "Dong_group_Sirius",
    "Dong_group_Sonic",
    "Dong_group_Spacy",
    "Dong_group_Sport / Xipo",
    "Dong_group_Sprint",
    "Dong_group_Vario",
    "Dong_group_Vespa",
    "Dong_group_Vision",
    "Dong_group_Wave",
    "Dong_group_Win",
    "Dong_group_Winner",
    "Dong_group_Winner X",
]

# ---------------------------------------------------------
# 4. DANH SÁCH DÒNG XE THEO HÃNG (DYNAMIC MAPPING)
# ---------------------------------------------------------
BRAND_MODELS_MAP = {
    "Honda": [
        "Air Blade",
        "SH",
        "SH Mode",
        "Vision",
        "Wave",
        "Future",
        "Winner",
        "Winner X",
        "Lead",
        "Blade",
        "CB",
        "CBR",
        "Click",
        "Cub",
        "Dream",
        "Dylan",
        "MSX 125",
        "PCX",
        "PS",
        "Sonic",
        "Spacy",
        "Win",
        "@",
        "Khác",
    ],
    "Yamaha": [
        "Exciter",
        "Sirius",
        "Janus",
        "Grande",
        "Jupiter",
        "NVX",
        "Nozza",
        "Nouvo",
        "Luvias",
        "FZ",
        "Mio",
        "PG-1",
        "R",
        "Khác",
    ],
    "Piaggio": ["Vespa", "Sprint", "Liberty", "GTS", "LX", "Khác"],
    "Suzuki": ["Satria", "Raider", "GSX", "Hayate", "Sport / Xipo", "Khác"],
    "SYM": ["Attila", "Elizabeth", "Elegant", "Shark", "Khác"],
    "Kawasaki": ["Khác"],
    "Ducati": ["Khác"],
    "Kymco": ["Khác"],
    "Khác": ["Khác"],
}

ORIGINS = [
    "Việt Nam",
    "Nhật Bản",
    "Thái Lan",
    "Đài Loan",
    "Trung Quốc",
    "Đức",
    "Mỹ",
    "Ấn Độ",
    "Nước khác",
    "Không rõ",
]

# ---------------------------------------------------------
# 5. GIAO DIỆN NGUỜI DÙNG (UI)
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🏍️ HỆ THỐNG ĐỊNH GIÁ XE MÁY THÔNG MINH</h1>
        <p>Công cụ hỗ trợ Salon định giá mua/bán xe cũ dựa trên thuật toán AI XGBoost</p>
    </div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Quản lý Salon")
    st.info("💡 **Mẹo:** Chọn Hãng xe để danh sách Dòng xe tự động lọc tương ứng.")
    st.divider()
    st.metric(label="Mô hình dự đoán", value="XGBoost v2.0")

st.subheader("📝 Nhập Thông Tin Xe Máy Cần Định Giá")

col1, col2 = st.columns(2)

with col1:
    # 1. Chọn Hãng Xe
    hang_xe = st.selectbox("Hãng xe", list(BRAND_MODELS_MAP.keys()))

    # 2. Dòng Xe tự động lọc theo Hãng Xe vừa chọn (Không cần load lại trang)
    available_models = BRAND_MODELS_MAP.get(hang_xe, ["Khác"])
    dong_xe = st.selectbox("Dòng xe", available_models)

    loai_xe = st.radio("Loại xe", ["Tay ga", "Xe số", "Xe côn tay / Khác"], horizontal=True)
    nam_san_xuat = st.number_input("Năm sản xuất", min_value=1990, max_value=2026, value=2020)

with col2:
    so_km = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=25000, step=1000)
    xuat_xu = st.selectbox("Xuất xứ", ORIGINS)
    dung_tich = st.selectbox(
        "Phân khối / Dung tích",
        ["Dưới 50cc", "50cc - 115cc", "116cc - 175cc", "Trên 175cc (PKL)", "Không rõ"],
    )

st.divider()

# ---------------------------------------------------------
# 6. DỰ ĐOÁN GIÁ XE
# ---------------------------------------------------------
if st.button("🚀 TÍNH GIÁ XE NGAY", use_container_width=True, type="primary"):
    if model is None:
        st.error("Mô hình chưa sẵn sàng. Vui lòng kiểm tra file xgb_price_model.pkl trong thư mục models/.")
    else:
        # Khởi tạo vector 81 thuộc tính = 0.0
        input_data = {feat: 0.0 for feat in FEATURE_NAMES}

        # Biến số nguyên
        current_year = 2026
        tuoi_xe = max(0, current_year - nam_san_xuat)
        input_data["Tuoi_xe_final"] = float(tuoi_xe)
        input_data["Km_final"] = float(so_km)
        input_data["Km_lam_tron"] = float(round(so_km, -3))

        # Phân khúc dung tích
        dung_tich_map = {"Dưới 50cc": 1, "50cc - 115cc": 2, "116cc - 175cc": 3, "Trên 175cc (PKL)": 4}
        if dung_tich in dung_tich_map:
            input_data["Dung_tich_ord"] = float(dung_tich_map[dung_tich])
        else:
            input_data["Dung_tich_khong_ro"] = 1.0

        # One-hot Loại Xe
        if loai_xe == "Tay ga":
            input_data["Loại xe_Tay ga"] = 1.0
        elif loai_xe == "Xe số":
            input_data["Loại xe_Xe số"] = 1.0

        # One-hot Xuất xứ
        origin_feat_map = {
            "Việt Nam": "Xuat_xu_clean_Việt Nam",
            "Nhật Bản": "Xuat_xu_clean_Nhật Bản",
            "Thái Lan": "Xuat_xu_clean_Thái Lan",
            "Trung Quốc": "Xuat_xu_clean_Trung Quốc",
            "Đài Loan": "Xuat_xu_clean_Đài Loan",
            "Đức": "Xuat_xu_clean_Đức",
            "Mỹ": "Xuat_xu_clean_Mỹ",
            "Ấn Độ": "Xuat_xu_clean_Ấn Độ",
            "Nước khác": "Xuat_xu_clean_Nước khác",
            "Không rõ": "Xuat_xu_clean_Không rõ",
        }
        if xuat_xu in origin_feat_map and origin_feat_map[xuat_xu] in input_data:
            input_data[origin_feat_map[xuat_xu]] = 1.0

        # One-hot Hãng xe
        brand_feat = f"Hang_group_{hang_xe}"
        if brand_feat in input_data:
            input_data[brand_feat] = 1.0
        else:
            input_data["Hang_group_Khác"] = 1.0

        # One-hot Dòng xe
        dong_feat = f"Dong_group_{dong_xe}"
        if dong_feat in input_data:
            input_data[dong_feat] = 1.0
        else:
            input_data["Dong_group_Khác"] = 1.0

        # DataFrame chuẩn 81 cột đúng thứ tự
        df_input = pd.DataFrame([input_data])[FEATURE_NAMES]

        try:
            predicted_price = model.predict(df_input)[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>💰 Giá Dự Đoán Cho Xe {hang_xe} {dong_xe} ({nam_san_xuat}):</h3>
                    <div class="price-tag">{predicted_price:,.1f} TRIỆU VNĐ</div>
                    <p><i>* Mức giá ước tính dựa trên thị trường xe cũ hiện tại.</i></p>
                </div>
            """,
                unsafe_allow_html=True,
            )

            col_buy, col_sell = st.columns(2)
            with col_buy:
                st.metric("Gợi ý giá nhập (Mua vào)", f"{predicted_price * 0.88:,.1f} Tr VNĐ", "-12%")
            with col_sell:
                st.metric("Gợi ý giá niêm yết (Bán ra)", f"{predicted_price * 1.05:,.1f} Tr VNĐ", "+5%")

        except Exception as e:
            st.error(f"❌ Lỗi trong quá trình dự đoán dữ liệu: {e}")