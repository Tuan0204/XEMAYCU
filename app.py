import base64
from pathlib import Path
import random
import numpy as np
import pandas as pd
import streamlit as st

# ⚠️ LỆNH NÀY BẮT BUỘC PHẢI ĐẶT ĐẦU TIÊN
st.set_page_config(
    page_title="Motorbike Price & Anomaly", page_icon="🏍️", layout="wide"
)

try:
    from model_utils import (
        SEGMENT_DUNG_TICH_GOI_Y,
        build_features,
        classify_price,
        get_dong_options,
        get_hang_options_for_segment,
        get_options,
        load_model_bundle,
        predict_price,
    )
except Exception as e:
    st.error(f"Lỗi import model_utils: {e}")

ACCENT = "#E85D25"

# ============================================================
# HÀM CHUYỂN ĐỔI ẢNH BACKGROUND LOCAL SANG BASE64
# ============================================================
def get_base64_image(image_path: str) -> str:
    path = Path(__file__).parent / image_path
    if path.exists():
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# Thử nạp ảnh bg.png (đặt cùng thư mục với file app.py)
bg_b64 = get_base64_image("bg.png")
if bg_b64:
    bg_style = f"url('data:image/png;base64,{bg_b64}') no-repeat center center fixed !important; background-size: cover !important;"
else:
    # Dự phòng nếu chưa chép file bg.png vào thư mục
    bg_style = "linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%) !important;"

# ============================================================
# TỪ ĐIỂN MAPPING TỰ ĐỘNG TÊN CỘT DÙNG CHO BATCH PREDICT
# ============================================================
COLUMN_MAPPING = {
    "Hang_xe": "Hang_xe",
    "hang_xe": "Hang_xe",
    "thuong_hieu": "Hang_xe",
    "Thương hiệu": "Hang_xe",
    "Hang": "Hang_xe",
    "Dong_xe": "Dong_xe",
    "dong_xe": "Dong_xe",
    "Dòng xe": "Dong_xe",
    "Dong": "Dong_xe",
    "Loai_xe": "Loai_xe",
    "loai_xe": "Loai_xe",
    "Loại xe": "Loai_xe",
    "Xuat_xu": "Xuat_xu",
    "xuat_xu": "Xuat_xu",
    "Xuất xứ": "Xuat_xu",
    "Dung_tich": "Dung_tich",
    "dung_tich": "Dung_tich",
    "dung_tich_cc": "Dung_tich",
    "Dung tích xe": "Dung_tich",
    "Nam_dang_ky": "Nam_dang_ky",
    "nam_dang_ky": "Nam_dang_ky",
    "Năm đăng ký": "Nam_dang_ky",
    "Nam_san_xuat": "Nam_dang_ky",
    "Km": "Km",
    "so_km": "Km",
    "Số Km đã đi": "Km",
    "So_km": "Km",
    "Gia_rao": "Gia_rao",
    "gia_rao": "Gia_rao",
    "Giá": "Gia_rao",
    "Gia": "Gia_rao",
}


def clean_price(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    clean_str = (
        str(val)
        .replace("đ", "")
        .replace("Đ", "")
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "")
        .strip()
    )
    try:
        return float(clean_str)
    except ValueError:
        return 0.0


# ============================================================
# CSS GIAO DIỆN & BACKGROUND TECH VECTOR (TRẮNG/XÁM)
# ============================================================
st.markdown(
    f"""
<style>
    /* 1. NỀN TỔNG THỂ DÙNG ẢNH BG VECTOR CÔNG NGHỆ */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: {bg_style}
    }}
    
    /* 2. SIDEBAR CẢI TIẾN TRẮNG TINH NỔI BẬT */
    section[data-testid="stSidebar"] {{
        width: 320px !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        padding: 1rem 0.5rem;
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0 15px rgba(0,0,0,0.04);
    }}

    /* 3. HERO BANNER CỬA HÀNG / SHOWROOM XE MÁY */
    .hero-banner {{
        background: 
            linear-gradient(135deg, rgba(15, 23, 42, 0.78) 0%, rgba(232, 93, 37, 0.72) 100%), 
            url('https://images.unsplash.com/photo-1558981403-c5f9899a28bc?q=80&w=1600&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-radius: 20px;
        padding: 48px 25px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }}
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        color: #ffffff;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 14px;
        border: 1px solid rgba(255, 255, 255, 0.4);
    }}
    .hero-banner h1 {{
        color: #ffffff !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        margin-bottom: 12px !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }}
    .hero-banner p {{
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1rem !important;
        max-width: 680px;
        margin: 0 auto !important;
        text-shadow: 0 1px 4px rgba(0,0,0,0.4);
    }}

    /* 4. METRIC CARDS TRẮNG BO GÓC ĐỔ BÓNG NỔI TRÊN NỀN BG */
    div[data-testid="stMetric"] {{
        background-color: #ffffff !important;
        border-radius: 16px !important;
        padding: 18px 22px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #e2e8f0 !important;
    }}

    /* 5. KHUNG KẾT QUẢ VÀ CARD NỘI DUNG */
    .price-result {{ 
        border: 1px solid rgba(232,93,37,0.35); 
        background: #ffffff; 
        border-radius: 16px; 
        padding: 22px; 
        box-shadow: 0 8px 20px rgba(232, 93, 37, 0.08);
    }}
    .price-result .pr-label {{ font-size: 14px; color: #64748b; margin: 0; }}
    .price-result .pr-value {{ font-size: 2.2rem; font-weight: 800; color: {ACCENT}; margin: 4px 0 8px; }}
    .price-result .pr-note {{ font-size: 13px; color: #64748b; margin: 0; }}

    /* 6. NÚT BẤM & TABS */
    .stButton > button {{ 
        background-color: {ACCENT}; 
        color: white; 
        border: none; 
        border-radius: 10px; 
        font-weight: 600; 
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 12px rgba(232, 93, 37, 0.25);
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{ 
        background-color: #c44a1a; 
        color: white; 
        transform: translateY(-1px);
    }}
    div[data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {ACCENT} !important; font-weight: bold; }}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = Path(__file__).parent / "models" / "xgb_price_model.pkl"


@st.cache_resource
def get_bundle():
    return load_model_bundle(str(MODEL_PATH))


try:
    bundle = get_bundle()
    OPTS = get_options(bundle)
    MODEL_OK = True
    MODEL_ERROR = ""
except Exception as e:
    MODEL_OK = False
    MODEL_ERROR = str(e)


# ============================================================
# DATABASE GIẢ LẬP VỚI TRẠNG THÁI KIỂM DUYỆT TIN
# ============================================================
if "mock_db" not in st.session_state:
    np.random.seed(42)
    users = [
        "Nguyễn Văn An", "Trần Thị Bích", "Lê Hoàng Nam", "Phạm Minh Tuấn",
        "Vũ Thị Mai", "Đặng Quốc Bảo", "Bùi Phương Thảo", "Hoàng Anh Dũng",
        "Đỗ Thu Trang", "Ngô Văn Hùng",
    ]
    phones = [f"09{random.randint(10000000, 99999999)}" for _ in range(10)]
    vehicles = [
        ("Honda", "Wave Alpha", "Xe số", "100 - 150 cc", 18500000, 18000000),
        ("Honda", "SH 150i", "Tay ga", "100 - 150 cc", 82000000, 85000000),
        ("Yamaha", "Exciter 150", "Xe tay côn", "100 - 150 cc", 29000000, 31000000),
        ("Ducati", "Monster 821", "Tay côn / PKL", "Trên 175 cc", 220000000, 215000000),
        ("Piaggio", "Vespa LX", "Tay ga", "100 - 150 cc", 12000000, 25000000),
        ("Honda", "Vision", "Tay ga", "100 - 150 cc", 55000000, 32000000),
        ("Detech", "67", "Xe số", "50 - 100 cc", 14000000, 13500000),
        ("Yamaha", "Sirius", "Xe số", "100 - 150 cc", 11500000, 12000000),
        ("Kawasaki", "Z900", "Tay côn / PKL", "Trên 175 cc", 195000000, 190000000),
        ("Suzuki", "Satria 150", "Xe tay côn", "100 - 150 cc", 38000000, 37500000),
    ]
    
    data = []
    for i in range(10):
        u, p, v = users[i], phones[i], vehicles[i]
        gia_rao, gia_goi_y = v[4], v[5]
        nhan, chenh = classify_price(gia_rao, gia_goi_y, threshold=0.15)
        
        data.append({
            "Mã tin": f"TIN-20260801-{100+i}",
            "Người đăng": u,
            "Số điện thoại": p,
            "Hãng xe": v[0],
            "Dòng xe": v[1],
            "Loại xe": v[2],
            "Dung tích": v[3],
            "Giá rao (VNĐ)": gia_rao,
            "Giá đề xuất (VNĐ)": gia_goi_y,
            "Chênh lệch": f"{chenh:+.1f}%",
            "Đánh giá giá": "🔴 Quá đắt" if nhan == "Quá đắt" else ("🟡 Quá rẻ" if nhan == "Quá rẻ" else "🟢 Hợp lý"),
            "Trạng thái duyệt": "⏳ Chờ duyệt"
        })
    st.session_state.mock_db = pd.DataFrame(data)


# ============================================================
# FORM NHẬP THÔNG TIN XE
# ============================================================
def render_vehicle_input_form(key_prefix: str):
    segment = st.radio(
        "Phân khúc xe",
        ["Xe phổ thông", "Xe cổ điển", "Phân khối lớn"],
        horizontal=True,
        key=f"{key_prefix}_segment",
    )
    
    seg_key = f"{key_prefix}_{segment.replace(' ', '_')}"
    
    hang_options = get_hang_options_for_segment(bundle, segment, OPTS)
    all_dt_labels = OPTS["dung_tich_labels"]
    
    if segment == "Phân khối lớn":
        allowed_dt = [dt for dt in all_dt_labels if "Trên 175" in dt or "175" in dt]
        if not allowed_dt: allowed_dt = all_dt_labels
        default_dt_label = allowed_dt[0]
        default_hang_idx = hang_options.index("Ducati") if "Ducati" in hang_options else 0
        default_loai = "Xe tay côn" if "Xe tay côn" in OPTS["Loại xe"] else OPTS["Loại xe"][0]
    elif segment == "Xe cổ điển":
        allowed_dt = [dt for dt in all_dt_labels if "50" in dt or "100" in dt]
        if not allowed_dt: allowed_dt = all_dt_labels
        default_dt_label = allowed_dt[0]
        default_hang_idx = hang_options.index("Detech") if "Detech" in hang_options else 0
        default_loai = "Xe số" if "Xe số" in OPTS["Loại xe"] else OPTS["Loại xe"][0]
    else:
        allowed_dt = [dt for dt in all_dt_labels if "100" in dt or "150" in dt]
        if not allowed_dt: allowed_dt = all_dt_labels
        default_dt_label = allowed_dt[0]
        default_hang_idx = hang_options.index("Honda") if "Honda" in hang_options else 0
        default_loai = "Xe số" if "Xe số" in OPTS["Loại xe"] else OPTS["Loại xe"][0]

    col1, col2 = st.columns(2)
    with col1:
        hang = st.selectbox("Hãng xe *", hang_options, index=default_hang_idx, key=f"{seg_key}_hang")
    with col2:
        raw_dong_options = get_dong_options(bundle, hang, OPTS)
        if segment == "Phân khối lớn":
            pkl_keywords = ["cbr", "z", "monster", "panigale", "r1", "r3", "r6", "cb", "ninja", "rebel", "tmax", "versys", "duke"]
            filtered_dong = [d for d in raw_dong_options if any(k in d.lower() for k in pkl_keywords)]
            dong_options = filtered_dong if len(filtered_dong) > 0 else raw_dong_options
        elif segment == "Xe cổ điển":
            classic_keywords = ["67", "cub", "win", "vespa", "mobylette", "cd", "minsk"]
            filtered_dong = [d for d in raw_dong_options if any(k in d.lower() for k in classic_keywords)]
            dong_options = filtered_dong if len(filtered_dong) > 0 else raw_dong_options
        else:
            pkl_keywords = ["z1000", "z900", "monster", "panigale", "1000cc", "821"]
            filtered_dong = [d for d in raw_dong_options if not any(k in d.lower() for k in pkl_keywords)]
            dong_options = filtered_dong if len(filtered_dong) > 0 else raw_dong_options

        dong = st.selectbox("Dòng xe *", dong_options, key=f"{seg_key}_dong")

    col3, col4 = st.columns(2)
    with col3:
        loai_idx = OPTS["Loại xe"].index(default_loai) if default_loai in OPTS["Loại xe"] else 0
        loai_xe = st.selectbox("Loại xe *", OPTS["Loại xe"], index=loai_idx, key=f"{seg_key}_loai")
    with col4:
        xuat_xu = st.selectbox("Xuất xứ", OPTS["Xuat_xu_clean"], key=f"{seg_key}_xuatxu")

    col5, col6 = st.columns(2)
    with col5:
        dung_tich_unknown = st.checkbox("Không rõ dung tích xe", key=f"{seg_key}_dtkr")
        dt_idx = allowed_dt.index(default_dt_label) if default_dt_label in allowed_dt else 0
        dung_tich_label = st.selectbox("Dung tích xe *", allowed_dt, index=dt_idx, disabled=dung_tich_unknown, key=f"{seg_key}_dt")
    with col6:
        reg_before_1980 = st.checkbox("Xe đăng ký trước năm 1980", key=f"{seg_key}_reg1980")
        nam_dang_ky = st.number_input("Năm đăng ký *", min_value=1980, max_value=2026, value=2018, disabled=reg_before_1980, key=f"{seg_key}_nam")

    col7, col8 = st.columns(2)
    with col7:
        km_unknown = st.checkbox("Không rõ số Km đã đi", key=f"{seg_key}_kmkr")
        km = st.number_input("Số Km đã đi *", min_value=0, max_value=300000, value=26000, step=1000, disabled=km_unknown, key=f"{seg_key}_km")
    with col8:
        khu_vuc = st.selectbox(
            "Khu vực giao dịch (Quận/Huyện)",
            [f"Quận {i}" for i in range(1, 13)] + ["Bình Thạnh", "Gò Vấp", "Tân Bình", "Thủ Đức"],
            key=f"{seg_key}_khuvuc",
        )

    is_valid = True
    if segment == "Phân khối lớn":
        if "Trên 175" not in dung_tich_label and not dung_tich_unknown:
            st.error(f"⚠️ Xe PKL bắt buộc dung tích **Trên 175 cc**.")
            is_valid = False
        else:
            st.success(f"✅ Xác thực hợp lệ: **{hang} {dong}** ({dung_tich_label})")
    elif segment == "Xe cổ điển":
        if "Trên 175" in dung_tich_label:
            st.error(f"⚠️ Xe cổ điển không dùng dung tích PKL.")
            is_valid = False
        else:
            st.success(f"✅ Xác thực hợp lệ: **{hang} {dong}** ({dung_tich_label})")
    else:
        st.success(f"✅ Xác thực hợp lệ: **{hang} {dong}** ({dung_tich_label})")

    return dict(
        segment=segment,
        hang=hang,
        dong=dong,
        loai_xe=loai_xe,
        xuat_xu=xuat_xu,
        dung_tich_label=dung_tich_label,
        dung_tich_unknown=dung_tich_unknown,
        reg_before_1980=reg_before_1980,
        nam_dang_ky=int(nam_dang_ky),
        km=km,
        km_unknown=km_unknown,
        khu_vuc=khu_vuc,
        is_valid=is_valid,
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🏍️ Motorbike Price")
    page = st.radio(
        "Điều hướng",
        [
            "Trang chủ",
            "Business problem",
            "Phân công nhóm",
            "Người bán xe",
            "Admin",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 📌 **Nhóm 7**")
    st.markdown("""
    **Thành viên:**
    * Nguyễn Thị Thúy Hằng (Trưởng nhóm)
    * Lê Ngọc Tuấn
    * Nguyễn Nhật Minh Thư
    """)
    st.caption("Cập nhật: 08/2026")
    if not MODEL_OK:
        st.error("⚠️ Chưa load được model")

# ============================================================
# TRANG CHỦ
# ============================================================
if page == "Trang chủ":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">CHỢ TỐT MOTORBIKE INTELLIGENCE</div>
            <h1>🏍️ Hệ thống định giá & kiểm duyệt tin xe máy Chợ Tốt</h1>
            <p>Ứng dụng AI gợi ý giá bán cạnh tranh cho Người bán & Kiểm duyệt tin tự động cho Admin</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Tin đăng huấn luyện", "7,208")
    c2.metric("Số cột dữ liệu", "18")
    c3.metric("Bài toán song song", "2")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("ℹ️ Chọn vai trò của bạn ở menu bên trái: **Người bán xe** hoặc **Admin**.")

# ============================================================
# BUSINESS PROBLEM
# ============================================================
elif page == "Business problem":
    st.title("Business Problem")
    st.caption("Tổng quan về bài toán & nhu cầu thực tế")
    with st.expander("Vấn đề hiện tại", expanded=True):
        st.write(
            "Chợ Tốt hiện chưa có chức năng gợi ý giá bán và cảnh báo tin đăng bất thường "
            "cho xe máy cũ. Điều này ảnh hưởng đến cả người bán lẫn đội ngũ kiểm duyệt."
        )

    st.markdown("**Nhu cầu các bên liên quan**")
    with st.expander("🧑 Người bán"):
        st.write(
            'Cần "Price Engine" gợi ý giá cạnh tranh để bán nhanh, không bị thiệt ("hớ").'
        )
    with st.expander("🛡️ Admin / Nền tảng"):
        st.write(
            "Cần bộ lọc tự động phân loại Cờ xanh (hợp lệ) / Cờ đỏ (bất thường) để giảm tải kiểm duyệt thủ công."
        )

    st.markdown("**Dữ liệu**")
    with st.expander("data_motobikes.xlsx"):
        st.write(
            "7,208 tin đăng xe máy cũ tại Tp.HCM, 18 cột (dữ liệu bảo mật, chỉ dùng học tập/nghiên cứu)."
        )

# ============================================================
# PHÂN CÔNG NHÓM
# ============================================================
elif page == "Phân công nhóm":
    st.title("Phân công nhóm - Nhóm 7")
    df_team = pd.DataFrame(
        {
            "Thành viên": [
                "Nguyễn Thị Thúy Hằng",
                "Lê Ngọc Tuấn",
                "Nguyễn Nhật Minh Thư",
            ],
            "Vai trò": [
                "Trưởng nhóm",
                "Thành viên",
                "Thành viên",
            ],
            "Việc phụ trách": [
                "Bổ sung phân nhóm & test lại phần EDA, Bài toán 2 làm PySpark, Lên ý tưởng design GUI",
                "Bài toán 2 theo ML, Build GUI Streamlit",
                "Thực hiện phần EDA, Bài toán 1 theo 2 mô hình",
            ],
        }
    )
    st.table(df_team)

# ============================================================
# NGƯỜI BÁN XE
# ============================================================
elif page == "Người bán xe":
    st.title("🧑 Kênh Dành Cho Người Bán Xe")
    st.caption("Định giá, tham vấn giá muốn bán & hỗ trợ tạo nội dung tin đăng nhanh chóng")

    if not MODEL_OK:
        st.error(f"Không load được model: {MODEL_ERROR}")
        st.stop()

    tab_seller_eval, tab_seller_post = st.tabs([
        "💰 1. Thẩm định & Đánh giá giá muốn bán",
        "📝 2. Lấy thông tin & Mẫu đăng tin bài"
    ])

    vals = render_vehicle_input_form("seller_main")

    # Tính toán giá AI gợi ý
    feats = build_features(
        bundle, **{k: v for k, v in vals.items() if k not in ["segment", "khu_vuc", "is_valid"]}
    )
    gia_goi_y = predict_price(bundle, feats)
    gia_fmt_goi_y = f"{gia_goi_y:,.0f}".replace(",", ".")

    # --------------------------------------------------------
    # TAB 1: NHẬP GIÁ MONG MUỐN & ĐÁNH GIÁ (TRẢI ĐỀU FULL-WIDTH 100%)
    # --------------------------------------------------------
    with tab_seller_eval:
        st.subheader("Đánh giá mức giá bán mong muốn của bạn")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"""
            <div class="price-result">
                <p class="pr-label">💡 Giá gợi ý thị trường (AI dự báo)</p>
                <p class="pr-value">{gia_fmt_goi_y} VNĐ</p>
                <p class="pr-note">Mức giá chuẩn giúp tối ưu thời gian bán xe.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_p2:
            user_target_price = st.number_input(
                "Nhập mức giá bạn MONG MUỐN BÁN (VNĐ) *",
                min_value=1_000_000,
                max_value=1_000_000_000,
                value=int(np.round(gia_goi_y, -5)),
                step=500_000,
                key="user_target_price"
            )
            btn_eval = st.button("🚀 Chạy mô hình đánh giá mức giá", key="btn_eval_user_price")

        if btn_eval:
            if not vals["is_valid"]:
                st.error("⚠️ Thông tin xe chưa hợp lệ, vui lòng kiểm tra lại!")
            else:
                nhan, chenh_lech = classify_price(user_target_price, gia_goi_y, threshold=0.15)
                price_diff_val = user_target_price - gia_goi_y
                diff_fmt = f"{abs(price_diff_val):,.0f}".replace(",", ".")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Kết quả phân tích chi tiết từ AI")
                
                if nhan == "Quá đắt":
                    st.error(
                        f"🔴 **MỨC GIÁ CAO (Quá đắt so với thị trường)**\n\n"
                        f"* **Chênh lệch:** Giá bạn muốn bán **cao hơn {chenh_lech:.1f}%** (chênh khoảng +{diff_fmt} VNĐ) so với giá AI đề xuất.\n"
                        f"* **Lời khuyên:** Mức giá này có thể làm giảm khả năng tiếp cận người mua. Bạn nên cân nhắc điều chỉnh về sát mức **{gia_fmt_goi_y} VNĐ** để chốt đơn nhanh hơn."
                    )
                elif nhan == "Quá rẻ":
                    st.warning(
                        f"🟡 **MỨC GIÁ THẤP (Quá rẻ - Có thể bị hớ)**\n\n"
                        f"* **Chênh lệch:** Giá bạn muốn bán **thấp hơn {abs(chenh_lech):.1f}%** (chênh khoảng -{diff_fmt} VNĐ) so với giá AI đề xuất.\n"
                        f"* **Lời khuyên:** Bạn sẽ bán rất nhanh nhưng đang chịu **thiệt thòi về giá**. Bạn hoàn toàn có thể tự tin đặt giá mức **{gia_fmt_goi_y} VNĐ**."
                    )
                else:
                    st.success(
                        f"🟢 **MỨC GIÁ RẤT HỢP LÝ**\n\n"
                        f"* **Chênh lệch:** Mức giá bán chỉ chênh **{chenh_lech:+.1f}%** so với giá đề xuất của thị trường.\n"
                        f"* **Lời khuyên:** Mức giá có tính cạnh tranh cao, dễ dàng được duyệt và thanh khoản tốt!"
                    )

    # --------------------------------------------------------
    # TAB 2: TỰ ĐỘNG LẤY THÔNG TIN ĐỂ ĐĂNG TIN
    # --------------------------------------------------------
    with tab_seller_post:
        st.subheader("📋 Thông tin tổng hợp & Bài đăng mẫu")
        st.caption("Sao chép thông tin bên dưới để đăng bài nhanh chóng lên Chợ Tốt!")

        tieu_de_mau = f"Cần bán xe {vals['hang']} {vals['dong']} {vals['nam_dang_ky']} - Màu chuẩn đẹp"
        target_fmt = f"{user_target_price:,.0f}".replace(",", ".")
        km_fmt = f"{vals['km']:,}".replace(",", ".")

        mo_ta_mau = f"""Cần bán gấp xe {vals['hang']} {vals['dong']} đăng ký năm {vals['nam_dang_ky']}.
- Loại xe: {vals['loai_xe']} (Xuất xứ: {vals['xuat_xu']})
- Dung tích: {vals['dung_tich_label']}
- Số Km đã đi: {km_fmt} km
- Khu vực: {vals['khu_vuc']}, TP.HCM
- Giá bán mong muốn: {target_fmt} VNĐ (Có bớt chút lộc cho anh em thiện chí)
- Tình trạng xe còn rất mới, máy móc nguyên bản, chạy êm ái.
Liên hệ trực tiếp để xem xe và chạy thử!"""

        st.text_input("📌 Tiêu đề tin đăng mẫu:", value=tieu_de_mau)
        st.text_area("📄 Nội dung mô tả chi tiết mẫu:", value=mo_ta_mau, height=180)
        st.success("💡 Tip: Bạn có thể copy 2 đoạn thông tin trên dán trực tiếp vào trang đăng tin Chợ Tốt!")

# ============================================================
# ADMIN DASHBOARD
# ============================================================
elif page == "Admin":
    st.title("Dashboard Admin — Quản trị & Kiểm duyệt tin đăng")

    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "chotot123"

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        st.caption("Vui lòng đăng nhập tài khoản Quản trị viên")
        with st.form("login_form"):
            user = st.text_input("Tên đăng nhập", value=ADMIN_USERNAME)
            pw = st.text_input("Mật khẩu", type="password", value=ADMIN_PASSWORD)
            login_submitted = st.form_submit_button("Đăng nhập")

        if login_submitted:
            if user == ADMIN_USERNAME and pw == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("⚠️ Tên đăng nhập hoặc mật khẩu không chính xác!")
        st.stop()

    tab_overview, tab_admin_single, tab_admin_batch = st.tabs(
        [
            "📊 Overview & Duyệt tin đăng",
            "🕵️ Kiểm tra 1 tin đăng lẻ",
            "📁 Kiểm tra hàng loạt (File CSV/Excel)",
        ]
    )

    # --------------------------------------------------------
    # TAB 1: OVERVIEW & THÊM TÍNH NĂNG DUYỆT TIN
    # --------------------------------------------------------
    with tab_overview:
        st.subheader("Báo cáo tình hình & Phê duyệt tin đăng mới")
        
        df_db = st.session_state.mock_db
        
        total_tin = len(df_db)
        tin_hop_le = len(df_db[df_db["Đánh giá giá"] == "🟢 Hợp lý"])
        tin_bat_thuong = total_tin - tin_hop_le
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng tin mới đăng", f"{total_tin} tin")
        m2.metric("Tin hợp lệ (Cờ xanh)", f"{tin_hop_le}")
        m3.metric("Tin bất thường (Cờ đỏ)", f"{tin_bat_thuong}")
        m4.metric("Đã phê duyệt", f"{len(df_db[df_db['Trạng thái duyệt'] == '✅ Đã duyệt'])} tin")

        st.markdown("---")
        st.subheader("🛡️ Khu vực duyệt tin dành cho Admin")
        st.caption("Admin có thể xem thông tin và trực tiếp bấm **Duyệt tin** hoặc **Từ chối tin** bên dưới.")

        st.dataframe(df_db, use_container_width=True)

        st.markdown("#### ⚡ Xử lý kiểm duyệt tin lẻ:")
        c_sel, c_act1, c_act2 = st.columns([2, 1, 1])
        
        with c_sel:
            selected_tin = st.selectbox("Chọn Mã tin cần duyệt:", df_db["Mã tin"].tolist())
        
        with c_act1:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            if st.button("✅ Duyệt cho đăng tin", key="btn_approve"):
                st.session_state.mock_db.loc[st.session_state.mock_db["Mã tin"] == selected_tin, "Trạng thái duyệt"] = "✅ Đã duyệt"
                st.success(f"Đã duyệt cho phép đăng tin: **{selected_tin}**!")
                st.rerun()
                
        with c_act2:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            if st.button("❌ Từ chối đăng tin", key="btn_reject"):
                st.session_state.mock_db.loc[st.session_state.mock_db["Mã tin"] == selected_tin, "Trạng thái duyệt"] = "❌ Từ chối"
                st.error(f"Đã từ chối tin đăng: **{selected_tin}**!")
                st.rerun()

    # --------------------------------------------------------
    # TAB 2: PHÁT HIỆN BẤT THƯỜNG TỪNG XE
    # --------------------------------------------------------
    with tab_admin_single:
        st.caption("Nhập thông tin chi tiết xe và mức giá rao để hệ thống kiểm tra độ bất thường.")
        vals_admin = render_vehicle_input_form("admin_single")

        gia_rao_input = st.number_input(
            "Giá rao cần kiểm tra (VNĐ) *",
            min_value=0,
            step=1_000_000,
            value=45_000_000,
            key="gia_rao_admin_single",
        )

        if st.button("🔍 Kiểm tra độ bất thường", key="btn_check_admin_single"):
            if not vals_admin["is_valid"]:
                st.error("⚠️ Vui lòng điều chỉnh lại thông tin trước khi kiểm tra!")
            else:
                feats_admin = build_features(
                    bundle,
                    **{k: v for k, v in vals_admin.items() if k not in ["segment", "khu_vuc", "is_valid"]},
                )
                gia_du_doan_admin = predict_price(bundle, feats_admin)
                nhan, chenh_lech = classify_price(gia_rao_input, gia_du_doan_admin)
                gia_fmt_admin = f"{gia_du_doan_admin:,.0f}".replace(",", ".")

                if nhan == "Quá đắt":
                    st.error(f"⚠️ Giá rao **cao hơn {chenh_lech:.0f}%** so với giá đề xuất ({gia_fmt_admin} VNĐ) — **Cờ đỏ (Quá đắt)**.")
                elif nhan == "Quá rẻ":
                    st.warning(f"🔎 Giá rao **thấp hơn {abs(chenh_lech):.0f}%** so với giá đề xuất ({gia_fmt_admin} VNĐ) — **Cờ đỏ (Quá rẻ)**.")
                else:
                    st.success(f"✅ Giá rao **hợp lý**, chênh lệch {chenh_lech:.0f}% so với giá đề xuất ({gia_fmt_admin} VNĐ).")

    # --------------------------------------------------------
    # TAB 3: KIỂM TRA HÀNG LOẠT FILE CSV/EXCEL
    # --------------------------------------------------------
    with tab_admin_batch:
        st.caption("Tải lên file danh sách tin đăng để tự động kiểm tra giá và gắn cờ bất thường.")

        threshold_pct = (
            st.slider("Ngưỡng chênh lệch coi là bất thường (%)", 5, 50, 15, key="thr_admin") / 100
        )

        file = st.file_uploader(
            "File CSV/Excel (Tự động nhận diện tên cột tiếng Việt/Anh)",
            type=["csv", "xlsx"],
            key="upload_admin",
        )

        if file is not None:
            df_raw = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            df = df_raw.rename(columns=COLUMN_MAPPING)
            st.write(f"📋 **Dữ liệu gốc đã tải lên:** ({len(df_raw):,} dòng)")
            st.dataframe(df_raw, use_container_width=True)

            if st.button("🚩 Chạy dự báo & Kiểm tra hàng loạt", key="btn_run_admin_batch"):
                required = ["Hang_xe", "Dong_xe", "Loai_xe", "Xuat_xu", "Dung_tich", "Nam_dang_ky", "Km", "Gia_rao"]
                missing = [c for c in required if c not in df.columns]

                if missing:
                    st.error(f"File thiếu các cột bắt buộc: {', '.join(missing)}")
                else:
                    def _predict_row(r):
                        nam_dk = int(r["Nam_dang_ky"]) if pd.notna(r["Nam_dang_ky"]) and str(r["Nam_dang_ky"]).isdigit() else 2018
                        so_km_val = float(r["Km"]) if pd.notna(r["Km"]) else 25000.0

                        feats = build_features(
                            bundle,
                            hang=r["Hang_xe"],
                            dong=r["Dong_xe"],
                            loai_xe=r["Loai_xe"],
                            xuat_xu=r["Xuat_xu"],
                            dung_tich_label=str(r["Dung_tich"]),
                            dung_tich_unknown=pd.isna(r["Dung_tich"]),
                            reg_before_1980=False,
                            nam_dang_ky=nam_dk,
                            km=so_km_val,
                            km_unknown=pd.isna(r["Km"]),
                        )
                        return predict_price(bundle, feats)

                    df["Gia_rao_clean"] = df["Gia_rao"].apply(clean_price)
                    df["gia_du_doan"] = df.apply(_predict_row, axis=1)

                    results = df.apply(
                        lambda r: classify_price(r["Gia_rao_clean"], r["gia_du_doan"], threshold=threshold_pct),
                        axis=1,
                    )
                    df["nhan_xet"] = results.apply(lambda t: t[0])
                    df["chenh_lech_%"] = results.apply(lambda t: round(t[1], 1))

                    flagged = df[df["nhan_xet"] != "Hợp lý"]

                    st.markdown("---")
                    st.subheader("Kết quả phân tích hàng loạt")
                    st.markdown(
                        f'<p style="font-size:2.5rem;font-weight:700;color:{ACCENT};margin:0 0 12px;">'
                        f'{len(flagged)} <span style="font-size:1.1rem;font-weight:400;color:#64748b;">'
                        f'tin đăng bất thường ({len(flagged) / max(len(df), 1) * 100:.1f}% tổng số tin)</span></p>',
                        unsafe_allow_html=True,
                    )

                    tab_all, tab_expensive, tab_cheap = st.tabs(["📊 Tất cả kết quả", "🔴 Cờ đỏ: Quá đắt", "🟡 Cờ đỏ: Quá rẻ"])
                    with tab_all: st.dataframe(df, use_container_width=True)
                    with tab_expensive: st.dataframe(df[df["nhan_xet"] == "Quá đắt"], use_container_width=True)
                    with tab_cheap: st.dataframe(df[df["nhan_xet"] == "Quá rẻ"], use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("⬇️ Tải kết quả CSV", data=csv, file_name="ket_qua_kiem_tra_admin.csv", mime="text/csv", key="dl_admin_batch")

    st.markdown("---")
    if st.button("Đăng xuất", key="btn_logout"):
        st.session_state.admin_logged_in = False
        st.rerun()