from pathlib import Path
import pandas as pd
import streamlit as st

# ⚠️ LỆNH NÀY BẮT BUỘC PHẢI ĐẶT ĐẦU TIÊN TRƯỚC KHI IMPORT MODEL UTILS
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
    # Hãng xe
    "Hang_xe": "Hang_xe",
    "hang_xe": "Hang_xe",
    "thuong_hieu": "Hang_xe",
    "Thương hiệu": "Hang_xe",
    "Hang": "Hang_xe",
    # Dòng xe
    "Dong_xe": "Dong_xe",
    "dong_xe": "Dong_xe",
    "Dòng xe": "Dong_xe",
    "Dong": "Dong_xe",
    # Loại xe
    "Loai_xe": "Loai_xe",
    "loai_xe": "Loai_xe",
    "Loại xe": "Loai_xe",
    # Xuất xứ
    "Xuat_xu": "Xuat_xu",
    "xuat_xu": "Xuat_xu",
    "Xuất xứ": "Xuat_xu",
    # Dung tích
    "Dung_tich": "Dung_tich",
    "dung_tich": "Dung_tich",
    "dung_tich_cc": "Dung_tich",
    "Dung tích xe": "Dung_tich",
    # Năm đăng ký
    "Nam_dang_ky": "Nam_dang_ky",
    "nam_dang_ky": "Nam_dang_ky",
    "Năm đăng ký": "Nam_dang_ky",
    "Nam_san_xuat": "Nam_dang_ky",
    # Số Km
    "Km": "Km",
    "so_km": "Km",
    "Số Km đã đi": "Km",
    "So_km": "Km",
    # Giá rao
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
    section[data-testid="stSidebar"] {{ background-color: #f0f2f6; }}
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
# HÀM HỖ TRỢ DÙNG CHUNG CHO FORM NHẬP THÔNG TIN XE
# (ĐÃ CHUYỂN LÊN ĐÂY ĐỂ KHÔNG ĐÁNH TAN CHUỖI IF/ELIF)
# ============================================================
def render_vehicle_input_form(key_prefix: str):
    segment = st.radio(
        "Phân khúc xe",
        ["Xe phổ thông", "Xe cổ điển", "Phân khối lớn"],
        horizontal=True,
        key=f"{key_prefix}_segment",
    )
    if segment == "Phân khối lớn":
        st.caption(
            "⚠️ Xe phân khối lớn có ít dữ liệu huấn luyện hơn — giá dự đoán chỉ mang tính tham khảo."
        )
    elif segment == "Xe cổ điển":
        st.caption("Gồm các dòng xe cổ như Detech 67, Cub, Dream, Win, Chaly...")

    hang_options = get_hang_options_for_segment(bundle, segment, OPTS)

    col1, col2 = st.columns(2)
    with col1:
        hang = st.selectbox("Hãng xe *", hang_options, key=f"{key_prefix}_hang")
    with col2:
        dong_options = get_dong_options(bundle, hang, OPTS)
        dong = st.selectbox("Dòng xe *", dong_options, key=f"{key_prefix}_dong")

    col3, col4 = st.columns(2)
    with col3:
        loai_xe = st.selectbox(
            "Loại xe *", OPTS["Loại xe"], key=f"{key_prefix}_loai"
        )
    with col4:
        xuat_xu = st.selectbox(
            "Xuất xứ", OPTS["Xuat_xu_clean"], key=f"{key_prefix}_xuatxu"
        )

    col5, col6 = st.columns(2)
    with col5:
        dung_tich_unknown = st.checkbox(
            "Không rõ dung tích xe", key=f"{key_prefix}_dtkr"
        )
        dung_tich_default = SEGMENT_DUNG_TICH_GOI_Y.get(
            segment, OPTS["dung_tich_labels"][0]
        )
        dt_idx = (
            OPTS["dung_tich_labels"].index(dung_tich_default)
            if dung_tich_default in OPTS["dung_tich_labels"]
            else 0
        )
        dung_tich_label = st.selectbox(
            "Dung tích xe",
            OPTS["dung_tich_labels"],
            index=dt_idx,
            disabled=dung_tich_unknown,
            key=f"{key_prefix}_dt",
        )
    with col6:
        reg_before_1980 = st.checkbox(
            "Xe đăng ký trước năm 1980", key=f"{key_prefix}_reg1980"
        )
        nam_dang_ky = st.number_input(
            "Năm đăng ký *",
            min_value=1980,
            max_value=2026,
            value=2018,
            disabled=reg_before_1980,
            key=f"{key_prefix}_nam",
        )

    col7, col8 = st.columns(2)
    with col7:
        km_unknown = st.checkbox(
            "Không rõ số Km đã đi", key=f"{key_prefix}_kmkr"
        )
        km = st.number_input(
            "Số Km đã đi *",
            min_value=0,
            max_value=300000,
            value=26000,
            step=1000,
            disabled=km_unknown,
            key=f"{key_prefix}_km",
        )
    with col8:
        khu_vuc = st.selectbox(
            "Khu vực giao dịch (Quận/Huyện)",
            [f"Quận {i}" for i in range(1, 13)]
            + ["Bình Thạnh", "Gò Vấp", "Tân Bình", "Thủ Đức"],
            key=f"{key_prefix}_khuvuc",
        )

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
    st.caption("Nhóm DL07_K314  \nĐH KHTN Tp.HCM  \nCập nhật: 08/2026")
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
    c1.metric("Tin đăng", "7,208")
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
elif page == "Phân công nhóm":
    st.title("Phân công nhóm")
    df_team = pd.DataFrame(
        {
            "Thành viên": ["Thành viên A", "Thành viên B"],
            "Việc phụ trách": [
                "Price Prediction, GUI người bán xe",
                "Anomaly Detection, GUI Admin",
            ],
        }
    )
    st.table(df_team)
    st.caption("Liên hệ: nhom.dl07k314@example.com")

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
        feats = build_features(
            bundle, **{k: v for k, v in vals.items() if k != "khu_vuc"}
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
    st.title("Dashboard Admin — Phát hiện bất thường & Kiểm tra")

    # MẬT KHẨU CỐ ĐỊNH CỦA ADMIN
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "chotot123"

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    # Giao diện Đăng nhập
    if not st.session_state.admin_logged_in:
        st.caption("Vui lòng đăng nhập tài khoản Quản trị viên")
        with st.form("login_form"):
            user = st.text_input("Tên đăng nhập", placeholder="admin")
            pw = st.text_input(
                "Mật khẩu", type="password", placeholder="••••••••"
            )
            login_submitted = st.form_submit_button("Đăng nhập")

        if login_submitted:
            if user == ADMIN_USERNAME and pw == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("⚠️ Tên đăng nhập hoặc mật khẩu không chính xác!")
        st.stop()

    if not MODEL_OK:
        st.error(f"Không load được model: {MODEL_ERROR}")
        st.stop()

    tab_admin_single, tab_admin_batch = st.tabs(
        [
            "🕵️ Kiểm tra 1 tin đăng bất thường",
            "📁 Kiểm tra hàng loạt (File CSV/Excel)",
        ]
    )

    # --------------------------------------------------------
    # TAB 1: PHÁT HIỆN BẤT THƯỜNG TỪNG XE
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
            feats_admin = build_features(
                bundle,
                **{k: v for k, v in vals_admin.items() if k != "khu_vuc"},
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
    # TAB 2: KIỂM TRA HÀNG LOẠT BẰNG FILE
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