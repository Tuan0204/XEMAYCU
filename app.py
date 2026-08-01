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
# CSS GIAO DIỆN
# ============================================================
st.markdown(
    f"""
<style>
    section[data-testid="stSidebar"] {{
        width: 320px !important;
        background-color: #f8f9fa;
        padding: 1rem 0.5rem;
    }}
    section[data-testid="stSidebar"] .stRadio > div {{
        gap: 8px;
    }}
    .stButton > button {{ background-color: {ACCENT}; color: white; border: none; border-radius: 8px; font-weight: 600; }}
    .stButton > button:hover {{ background-color: #c44a1a; color: white; }}
    div[data-baseweb="tab-highlight"] {{ background-color: {ACCENT} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {ACCENT} !important; }}
    .price-result {{ border: 1px solid rgba(232,93,37,0.35); background: #fff6f0; border-radius: 10px; padding: 18px 20px; margin-top: 8px; }}
    .price-result .pr-label {{ font-size: 13px; color: rgb(120,122,132); margin: 0; }}
    .price-result .pr-value {{ font-size: 1.9rem; font-weight: 700; color: {ACCENT}; margin: 4px 0 8px; }}
    .price-result .pr-note {{ font-size: 12px; color: rgb(120,122,132); margin: 0; }}
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
# TẠO DATABASE GIẢ LẬP HỆ THỐNG MỚI ĐĂNG (TAB OVERVIEW ADMIN)
# ============================================================
@st.cache_data
def load_mock_recent_database():
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
            "Trạng thái": "🔴 Quá đắt" if nhan == "Quá đắt" else ("🟡 Quá rẻ" if nhan == "Quá rẻ" else "🟢 Hợp lý")
        })
    return pd.DataFrame(data)


# ============================================================
# FORM NHẬP THÔNG TIN XE (ĐÃ LỌC HÃNG & DÒNG XE THEO PHÂN KHÚC)
# ============================================================
def render_vehicle_input_form(key_prefix: str):
    segment = st.radio(
        "Phân khúc xe",
        ["Xe phổ thông", "Xe cổ điển", "Phân khối lớn"],
        horizontal=True,
        key=f"{key_prefix}_segment",
    )
    
    seg_key = f"{key_prefix}_{segment.replace(' ', '_')}"
    
    # 1. LỌC DANH SÁCH HÃNG XE THEO PHÂN KHÚC
    hang_options = get_hang_options_for_segment(bundle, segment, OPTS)
    
    # 2. LỌC DANH SÁCH DUNG TÍCH PHÙ HỢP THEO PHÂN KHÚC
    all_dt_labels = OPTS["dung_tich_labels"]
    if segment == "Phân khối lớn":
        allowed_dt = [dt for dt in all_dt_labels if "Trên 175" in dt or "175" in dt]
        if not allowed_dt:
            allowed_dt = all_dt_labels
        default_dt_label = allowed_dt[0]
        default_hang_idx = hang_options.index("Ducati") if "Ducati" in hang_options else 0
        default_loai = "Xe tay côn" if "Xe tay côn" in OPTS["Loại xe"] else OPTS["Loại xe"][0]
    elif segment == "Xe cổ điển":
        allowed_dt = [dt for dt in all_dt_labels if "50" in dt or "100" in dt]
        if not allowed_dt:
            allowed_dt = all_dt_labels
        default_dt_label = allowed_dt[0]
        default_hang_idx = hang_options.index("Detech") if "Detech" in hang_options else 0
        default_loai = "Xe số" if "Xe số" in OPTS["Loại xe"] else OPTS["Loại xe"][0]
    else: # Xe phổ thông
        allowed_dt = [dt for dt in all_dt_labels if "100" in dt or "150" in dt]
        if not allowed_dt:
            allowed_dt = all_dt_labels
        default_dt_label = allowed_dt[0]
        default_hang_idx = hang_options.index("Honda") if "Honda" in hang_options else 0
        default_loai = "Xe số" if "Xe số" in OPTS["Loại xe"] else OPTS["Loại xe"][0]

    col1, col2 = st.columns(2)
    with col1:
        hang = st.selectbox(
            "Hãng xe *", hang_options, index=default_hang_idx, key=f"{seg_key}_hang"
        )
    with col2:
        # LỌC DÒNG XE TƯƠNG ỨNG VỚI HÃNG VÀ PHÂN KHÚC
        raw_dong_options = get_dong_options(bundle, hang, OPTS)
        
        # BỘ LỌC TỰ ĐỘNG LỌC DÒNG XE THEO TÊN/PHÂN KHÚC
        if segment == "Phân khối lớn":
            # Ưu tiên các dòng xe PKL đặc trưng nếu hãng là Honda/Yamaha/Suzuki
            pkl_keywords = ["cbr", "z", "monster", "panigale", "r1", "r3", "r6", "cb", "ninja", "rebel", "tmax", "versys", "duke"]
            filtered_dong = [d for d in raw_dong_options if any(k in d.lower() for k in pkl_keywords)]
            dong_options = filtered_dong if len(filtered_dong) > 0 else raw_dong_options
        elif segment == "Xe cổ điển":
            classic_keywords = ["67", "cub", "win", "vespa", "mobylette", "cd", "minsk"]
            filtered_dong = [d for d in raw_dong_options if any(k in d.lower() for k in classic_keywords)]
            dong_options = filtered_dong if len(filtered_dong) > 0 else raw_dong_options
        else: # Xe phổ thông: Loại bỏ bớt các dòng xe PKL rõ rệt
            pkl_keywords = ["z1000", "z900", "monster", "panigale", "1000cc", "821"]
            filtered_dong = [d for d in raw_dong_options if not any(k in d.lower() for k in pkl_keywords)]
            dong_options = filtered_dong if len(filtered_dong) > 0 else raw_dong_options

        dong = st.selectbox("Dòng xe *", dong_options, key=f"{seg_key}_dong")

    col3, col4 = st.columns(2)
    with col3:
        loai_idx = OPTS["Loại xe"].index(default_loai) if default_loai in OPTS["Loại xe"] else 0
        loai_xe = st.selectbox(
            "Loại xe *", OPTS["Loại xe"], index=loai_idx, key=f"{seg_key}_loai"
        )
    with col4:
        xuat_xu = st.selectbox(
            "Xuất xứ", OPTS["Xuat_xu_clean"], key=f"{seg_key}_xuatxu"
        )

    col5, col6 = st.columns(2)
    with col5:
        dung_tich_unknown = st.checkbox(
            "Không rõ dung tích xe", key=f"{seg_key}_dtkr"
        )
        dt_idx = allowed_dt.index(default_dt_label) if default_dt_label in allowed_dt else 0
        dung_tich_label = st.selectbox(
            "Dung tích xe *",
            allowed_dt,
            index=dt_idx,
            disabled=dung_tich_unknown,
            key=f"{seg_key}_dt",
        )
    with col6:
        reg_before_1980 = st.checkbox(
            "Xe đăng ký trước năm 1980", key=f"{seg_key}_reg1980"
        )
        nam_dang_ky = st.number_input(
            "Năm đăng ký *",
            min_value=1980,
            max_value=2026,
            value=2018,
            disabled=reg_before_1980,
            key=f"{seg_key}_nam",
        )

    col7, col8 = st.columns(2)
    with col7:
        km_unknown = st.checkbox(
            "Không rõ số Km đã đi", key=f"{seg_key}_kmkr"
        )
        km = st.number_input(
            "Số Km đã đi *",
            min_value=0,
            max_value=300000,
            value=26000,
            step=1000,
            disabled=km_unknown,
            key=f"{seg_key}_km",
        )
    with col8:
        khu_vuc = st.selectbox(
            "Khu vực giao dịch (Quận/Huyện)",
            [f"Quận {i}" for i in range(1, 13)]
            + ["Bình Thạnh", "Gò Vấp", "Tân Bình", "Thủ Đức"],
            key=f"{seg_key}_khuvuc",
        )

    # ============================================================
    # REAL-TIME VALIDATION & HỘP THÔNG BÁO CẢNH BÁO PHÂN KHÚC
    # ============================================================
    is_valid = True
    
    if segment == "Phân khối lớn":
        if "Trên 175" not in dung_tich_label and not dung_tich_unknown:
            st.error(f"⚠️ **Cảnh báo không hợp lệ:** Xe phân khối lớn bắt buộc phải có dung tích **Trên 175 cc** (Bạn đang chọn: {dung_tich_label}).")
            is_valid = False
        else:
            st.success(f"✅ Đã xác thực chuẩn Phân khúc PKL: **{hang} {dong}** ({dung_tich_label})")

    elif segment == "Xe cổ điển":
        if "Trên 175" in dung_tich_label:
            st.error(f"⚠️ **Cảnh báo không hợp lệ:** Xe cổ điển không dùng dung tích Phân khối lớn ({dung_tich_label}). Vui lòng chọn lại!")
            is_valid = False
        else:
            st.success(f"✅ Đã xác thực chuẩn Phân khúc Xe cổ điển: **{hang} {dong}** ({dung_tich_label})")

    else:
        if "Trên 175" in dung_tich_label:
            st.warning(f"💡 **Lưu ý:** Bạn đang chọn dung tích {dung_tich_label} cho Xe phổ thông. Nếu đây là xe PKL, vui lòng chuyển Phân khúc sang **Phân khối lớn**.")
        else:
            st.success(f"✅ Đã xác thực chuẩn Phân khúc Xe phổ thông: **{hang} {dong}** ({dung_tich_label})")

    return dict(
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
    st.markdown("## 🏍️ Motorbike Price & Anomaly")
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
    * Lê Ngọc Tuấn
    * Nguyễn Nhật Minh Thư
    * Nguyễn Thị Thúy Hằng
    """)
    st.caption("Cập nhật: 08/2026")
    if not MODEL_OK:
        st.error("⚠️ Chưa load được model")

# ============================================================
# TRANG CHỦ
# ============================================================
if page == "Trang chủ":
    st.title("Motorbike Price & Anomaly Detection")
    st.caption(
        "Định giá xe máy cũ cho người bán, phát hiện tin đăng bất thường cho Admin Chợ Tốt"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Tin đăng huấn luyện", "7,208")
    c2.metric("Số cột dữ liệu", "18")
    c3.metric("Bài toán song song", "2")

    st.info(
        "ℹ️ Chọn vai trò của bạn ở menu bên trái: **Người bán xe** hoặc **Admin**."
    )

# ============================================================
# BUSINESS PROBLEM
# ============================================================
elif page == "Business problem":
    st.title("Business problem")
    st.caption("Vấn đề")
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
# NGƯỜI BÁN XE (Định giá xe lẻ)
# ============================================================
elif page == "Người bán xe":
    st.title("Định giá xe máy của bạn")
    st.caption(
        "Chọn phân khúc xe, nhập thông tin chi tiết để nhận giá tham khảo từ model XGBoost."
    )

    if not MODEL_OK:
        st.error(f"Không load được model: {MODEL_ERROR}")
        st.stop()

    vals = render_vehicle_input_form("seller")

    if st.button("💰 Định giá xe ngay", key="btn_predict_single"):
        if not vals["is_valid"]:
            st.error("⚠️ Vui lòng điều chỉnh lại thông tin Dung tích/Dòng xe cho đúng Phân khúc trước khi định giá!")
        else:
            feats = build_features(
                bundle, **{k: v for k, v in vals.items() if k not in ["khu_vuc", "is_valid"]}
            )
            gia_du_doan = predict_price(bundle, feats)
            gia_fmt = f"{gia_du_doan:,.0f}".replace(",", ".")
            st.markdown(
                f"""
            <div class="price-result">
                <p class="pr-label">Giá đề xuất cho {vals['hang']} {vals['dong']}</p>
                <p class="pr-value">{gia_fmt} VNĐ</p>
                <p class="pr-note">Dự đoán bởi XGBoost dựa trên thông tin bạn nhập.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

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
            # 🟢 TỰ ĐỘNG ĐIỀN TÊN VÀ MẬT KHẨU SẴN VÀO TẢI KHOẢN
            user = st.text_input("Tên đăng nhập", value=ADMIN_USERNAME)
            pw = st.text_input(
                "Mật khẩu", type="password", value=ADMIN_PASSWORD
            )
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
            "📊 Overview (Tổng quan ngày qua)",
            "🕵️ Kiểm tra 1 tin đăng lẻ",
            "📁 Kiểm tra hàng loạt (File CSV/Excel)",
        ]
    )

    # --------------------------------------------------------
    # TAB 1: OVERVIEW & DATABASE TIN ĐĂNG MỚI
    # --------------------------------------------------------
    with tab_overview:
        st.subheader("Báo cáo tình hình tin đăng (24h qua)")
        
        df_recent = load_mock_recent_database()
        
        total_tin = len(df_recent)
        tin_hop_le = len(df_recent[df_recent["Trạng thái"] == "🟢 Hợp lý"])
        tin_bat_thuong = total_tin - tin_hop_le
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng tin mới đăng", f"{total_tin} tin")
        m2.metric("Tin hợp lệ (Cờ xanh)", f"{tin_hop_le}", f"{tin_hop_le/total_tin*100:.0f}%")
        m3.metric("Tin bất thường (Cờ đỏ)", f"{tin_bat_thuong}", f"{tin_bat_thuong/total_tin*100:.0f}%", delta_color="inverse")
        m4.metric("Tỷ lệ rủi ro tin ảo", f"{tin_bat_thuong/total_tin*100:.1f}%")

        st.markdown("---")
        st.subheader("🗄️ Cơ sở dữ liệu tin đăng mới nhất & Chi tiết người đăng")
        st.caption("Danh sách người dùng vừa đăng tin trên hệ thống. Admin có thể tra cứu chính xác tên, số điện thoại và cờ cảnh báo.")
        
        filter_status = st.multiselect(
            "Lọc theo trạng thái:",
            ["🟢 Hợp lý", "🔴 Quá đắt", "🟡 Quá rẻ"],
            default=["🟢 Hợp lý", "🔴 Quá đắt", "🟡 Quá rẻ"]
        )
        
        df_filtered = df_recent[df_recent["Trạng thái"].isin(filter_status)]
        st.dataframe(df_filtered, use_container_width=True)

    # --------------------------------------------------------
    # TAB 2: PHÁT HIỆN BẤT THƯỜNG TỪNG XE (ĐÃ THÊM VALIDATION)
    # --------------------------------------------------------
    with tab_admin_single:
        st.caption(
            "Nhập thông tin chi tiết xe và mức giá rao để hệ thống kiểm tra độ bất thường so với giá gợi ý."
        )
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
                st.error("⚠️ Vui lòng điều chỉnh lại thông tin Dung tích/Dòng xe cho đúng Phân khúc trước khi kiểm tra!")
            else:
                feats_admin = build_features(
                    bundle,
                    **{k: v for k, v in vals_admin.items() if k not in ["khu_vuc", "is_valid"]},
                )
                gia_du_doan_admin = predict_price(bundle, feats_admin)
                nhan, chenh_lech = classify_price(gia_rao_input, gia_du_doan_admin)
                gia_fmt_admin = f"{gia_du_doan_admin:,.0f}".replace(",", ".")

                if nhan == "Quá đắt":
                    st.error(
                        f"⚠️ Giá rao **cao hơn khoảng {chenh_lech:.0f}%** so với giá đề xuất ({gia_fmt_admin} VNĐ) — **Cờ đỏ (Quá đắt)**."
                    )
                elif nhan == "Quá rẻ":
                    st.warning(
                        f"🔎 Giá rao **thấp hơn khoảng {abs(chenh_lech):.0f}%** so với giá đề xuất ({gia_fmt_admin} VNĐ) — **Cờ đỏ (Quá rẻ)**."
                    )
                else:
                    st.success(
                        f"✅ Giá rao **hợp lý**, chênh lệch khoảng {chenh_lech:.0f}% so với giá đề xuất ({gia_fmt_admin} VNĐ)."
                    )

    # --------------------------------------------------------
    # TAB 3: KIỂM TRA HÀNG LOẠT FILE CSV/EXCEL
    # --------------------------------------------------------
    with tab_admin_batch:
        st.caption(
            "Tải lên file danh sách tin đăng để hệ thống tự động kiểm tra giá dự báo và gắn cờ bất thường."
        )

        threshold_pct = (
            st.slider(
                "Ngưỡng chênh lệch coi là bất thường (%)",
                5,
                50,
                15,
                key="thr_admin",
            )
            / 100
        )

        file = st.file_uploader(
            "File CSV/Excel (Tự động nhận diện tên cột tiếng Việt/Anh)",
            type=["csv", "xlsx"],
            key="upload_admin",
        )

        if file is not None:
            df_raw = (
                pd.read_csv(file)
                if file.name.endswith(".csv")
                else pd.read_excel(file)
            )

            df = df_raw.rename(columns=COLUMN_MAPPING)

            st.write("Dữ liệu gốc đã tải lên:")
            st.dataframe(df_raw, use_container_width=True)

            if st.button(
                "🚩 Chạy dự báo & Kiểm tra hàng loạt", key="btn_run_admin_batch"
            ):
                required = [
                    "Hang_xe",
                    "Dong_xe",
                    "Loai_xe",
                    "Xuat_xu",
                    "Dung_tich",
                    "Nam_dang_ky",
                    "Km",
                    "Gia_rao",
                ]
                missing = [c for c in required if c not in df.columns]

                if missing:
                    st.error(
                        f"File thiếu các cột bắt buộc chưa thể nhận diện: {', '.join(missing)}"
                    )
                else:

                    def _predict_row(r):
                        nam_dk = (
                            int(r["Nam_dang_ky"])
                            if pd.notna(r["Nam_dang_ky"])
                            and str(r["Nam_dang_ky"]).isdigit()
                            else 2018
                        )
                        so_km_val = (
                            float(r["Km"]) if pd.notna(r["Km"]) else 25000.0
                        )

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
                        lambda r: classify_price(
                            r["Gia_rao_clean"],
                            r["gia_du_doan"],
                            threshold=threshold_pct,
                        ),
                        axis=1,
                    )
                    df["nhan_xet"] = results.apply(lambda t: t[0])
                    df["chenh_lech_%"] = results.apply(lambda t: round(t[1], 1))

                    flagged = df[df["nhan_xet"] != "Hợp lý"]

                    st.markdown("---")
                    st.subheader("Kết quả phân tích hàng loạt")

                    st.markdown(
                        f'<p style="font-size:2.5rem;font-weight:700;color:{ACCENT};margin:0 0 12px;">'
                        f'{len(flagged)} <span style="font-size:1.1rem;font-weight:400;color:rgb(120,122,132);">'
                        f'tin đăng bất thường ({len(flagged) / max(len(df), 1) * 100:.1f}% tổng số tin)</span></p>',
                        unsafe_allow_html=True,
                    )

                    tab_all, tab_expensive, tab_cheap = st.tabs(
                        [
                            "📊 Tất cả kết quả",
                            "🔴 Cờ đỏ: Quá đắt",
                            "🟡 Cờ đỏ: Quá rẻ",
                        ]
                    )
                    with tab_all:
                        st.dataframe(df, use_container_width=True)
                    with tab_expensive:
                        st.dataframe(
                            df[df["nhan_xet"] == "Quá đắt"],
                            use_container_width=True,
                        )
                    with tab_cheap:
                        st.dataframe(
                            df[df["nhan_xet"] == "Quá rẻ"],
                            use_container_width=True,
                        )

                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Tải kết quả CSV",
                        data=csv,
                        file_name="ket_qua_kiem_tra_admin.csv",
                        mime="text/csv",
                        key="dl_admin_batch",
                    )

    st.markdown("---")
    if st.button("Đăng xuất", key="btn_logout"):
        st.session_state.admin_logged_in = False
        st.rerun()