# -*- coding: utf-8 -*-
"""Giao dien Streamlit: dang tin ban xe may cu, du doan gia hop ly & phat hien tin bat thuong."""
import datetime
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "price_prediction_pipeline_v2.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "price_prediction_pipeline_v2_metrics.joblib")
ANOMALY_ARTIFACTS_PATH = os.path.join(BASE_DIR, "models", "anomaly_artifacts.joblib")
SAMPLE_BATCH_PATH = os.path.join(BASE_DIR, "sample_batch_upload.csv")

BATCH_REQUIRED_COLUMNS = ["thuong_hieu", "dong_xe", "loai_xe", "nam_dang_ky", "gia_rao"]
BATCH_OPTIONAL_COLUMNS = ["so_km", "dung_tich_cc", "xuat_xu", "quan", "tieu_de", "mo_ta"]
REFERENCE_YEAR = 2025  # nam tham chieu dung khi huan luyen model (du lieu thu thap den 07/2025)

# Trung binh / trung vi lay tu du lieu huan luyen, dung khi khong hoi nguoi dung (tieu de, mo ta)
DEFAULT_TITLE_LEN = 28
DEFAULT_DESC_LEN = 169
DEFAULT_KM = 10000
DEFAULT_DUNG_TICH = 125

DUNG_TICH_OPTIONS = {
    "Dưới 50 cc": 35.0,
    "50 - 100 cc": 75.0,
    "100 - 175 cc": 125,
    "Trên 175 cc": 250.0,
    "Không rõ": None,
}

LOAI_XE_OPTIONS = ["Xe số", "Tay ga", "Tay côn/Moto"]

XUAT_XU_OPTIONS = [
    "Việt Nam", "Nhật Bản", "Thái Lan", "Trung Quốc", "Đài Loan",
    "Hàn Quốc", "Mỹ", "Đức", "Ấn Độ", "Nước khác", "Không rõ",
]

QUAN_OPTIONS = [
    "Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8",
    "Quận 10", "Quận 11", "Quận 12", "Quận Bình Tân", "Quận Bình Thạnh",
    "Quận Gò Vấp", "Quận Phú Nhuận", "Quận Tân Bình", "Quận Tân Phú",
    "Huyện Bình Chánh", "Huyện Cần Giờ", "Huyện Củ Chi", "Huyện Hóc Môn",
    "Huyện Nhà Bè", "Thành phố Thủ Đức", "Không rõ",
]

BRAND_MODELS = {
    "Honda": ["SH", "SH Mode", "Air Blade", "Vision", "Lead", "PCX", "Vario", "Click",
              "Wave", "Future", "Blade", "Winner", "Winner X", "Sonic", "CB", "CBR",
              "Dream", "Cub", "Chaly", "Win", "67", "@", "Dylan", "PS", "Spacy", "Dòng khác", "Khác"],
    "Yamaha": ["Exciter", "Nvx", "Janus", "Grande", "Sirius", "Jupiter", "Mio", "Nouvo",
               "Nozza", "Luvias", "FZ", "R", "Dòng khác", "Khác"],
    "Piaggio": ["Vespa", "Liberty", "Sprint", "LX", "GTS", "Dòng khác", "Khác"],
    "Suzuki": ["Raider", "Satria", "GSX", "Hayate", "Sport / Xipo", "Dòng khác", "Khác"],
    "SYM": ["Attila", "Elegant", "Elizabeth", "Shark", "Dòng khác", "Khác"],
    "Kawasaki": ["Dòng khác", "Khác"],
    "Kymco": ["Dòng khác", "Khác"],
    "Ducati": ["Dòng khác", "Khác"],
    "KTM": ["Khác"],
    "Harley Davidson": ["Dòng khác", "Khác"],
    "Halim": ["Dòng khác"],
    "Daelim": ["Cub", "Dòng khác", "Khác"],
    "Detech": ["Dòng khác", "Khác"],
    "GPX": ["Khác"],
    "BMW": ["Dòng khác", "Khác"],
    "Hãng khác": ["Dòng khác"],
    "Khác": ["Cub", "Dòng khác", "Khác"],
}


@st.cache_resource
def load_price_model():
    model = joblib.load(MODEL_PATH)
    try:
        metrics = joblib.load(METRICS_PATH)
    except FileNotFoundError:
        metrics = None
    return model, metrics


@st.cache_resource
def load_anomaly_artifacts():
    try:
        return joblib.load(ANOMALY_ARTIFACTS_PATH)
    except FileNotFoundError:
        return None


def format_vnd(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{abs(x):,.0f} đ".replace(",", ".")


def percentile_rank(value: float, train_array) -> float:
    """% so tin trong tap huan luyen co gia tri tin hieu <= value (0-100)."""
    arr = np.asarray(train_array)
    return float((arr <= value).mean() * 100)


def render_vehicle_inputs(prefix: str):
    """Cac o nhap thong tin xe dung chung cho ca 2 tab (key rieng theo prefix de tranh trung ID)."""
    col1, col2 = st.columns(2)
    with col1:
        brands_sorted = sorted(BRAND_MODELS.keys())
        brand = st.selectbox("Thương hiệu *", brands_sorted, index=brands_sorted.index("Honda"), key=f"{prefix}_brand")
    with col2:
        model_options = BRAND_MODELS.get(brand, ["Khác"])
        dong_xe = st.selectbox("Dòng xe *", model_options, key=f"{prefix}_dong_xe")

    col3, col4 = st.columns(2)
    with col3:
        loai_xe = st.selectbox("Loại xe *", LOAI_XE_OPTIONS, key=f"{prefix}_loai_xe")
    with col4:
        dung_tich_label = st.selectbox("Dung tích xe", list(DUNG_TICH_OPTIONS.keys()), index=2, key=f"{prefix}_dung_tich")

    col5, col6 = st.columns(2)
    with col5:
        pre_1980 = st.checkbox("Xe đăng ký trước năm 1980", key=f"{prefix}_pre1980")
        nam_dang_ky = st.number_input(
            "Năm đăng ký *", min_value=1980, max_value=datetime.date.today().year,
            value=2018, step=1, disabled=pre_1980, key=f"{prefix}_nam",
        )
    with col6:
        km_unknown = st.checkbox("Không rõ số km đã đi", key=f"{prefix}_km_unk")
        so_km = st.number_input(
            "Số Km đã đi *", min_value=0, max_value=300_000, value=26_000, step=1_000,
            disabled=km_unknown, key=f"{prefix}_km",
        )

    col7, col8 = st.columns(2)
    with col7:
        xuat_xu = st.selectbox("Xuất xứ", XUAT_XU_OPTIONS, index=XUAT_XU_OPTIONS.index("Việt Nam"), key=f"{prefix}_xuatxu")
    with col8:
        quan = st.selectbox("Khu vực giao dịch (Quận/Huyện) *", QUAN_OPTIONS, key=f"{prefix}_quan")

    st.markdown("**Tin đăng (tuỳ chọn)**")
    tieu_de = st.text_input("Tiêu đề tin đăng", placeholder="VD: Bán Honda SH 150i 2018 chính chủ, xe đẹp", key=f"{prefix}_title")
    mo_ta = st.text_area("Mô tả chi tiết", placeholder="VD: Xe chạy êm, không lỗi, đầy đủ giấy tờ...", height=100, key=f"{prefix}_desc")

    return {
        "brand": brand, "dong_xe": dong_xe, "loai_xe": loai_xe, "dung_tich_label": dung_tich_label,
        "pre_1980": pre_1980, "nam_dang_ky": nam_dang_ky, "km_unknown": km_unknown, "so_km": so_km,
        "xuat_xu": xuat_xu, "quan": quan, "tieu_de": tieu_de, "mo_ta": mo_ta,
    }


def build_feature_row(vals: dict) -> pd.DataFrame:
    if vals["pre_1980"]:
        vehicle_age = REFERENCE_YEAR - 1979
        pre_1980_flag = 1
    else:
        vehicle_age = max(REFERENCE_YEAR - vals["nam_dang_ky"], 0)
        pre_1980_flag = 0

    if vals["km_unknown"]:
        so_km_da_di = DEFAULT_KM
        km_missing_flag = 1
    else:
        so_km_da_di = vals["so_km"]
        km_missing_flag = 0

    dung_tich_val = DUNG_TICH_OPTIONS[vals["dung_tich_label"]]
    if dung_tich_val is None:
        dung_tich_cc = DEFAULT_DUNG_TICH
        dung_tich_missing_flag = 1
    else:
        dung_tich_cc = dung_tich_val
        dung_tich_missing_flag = 0

    title_len = len(vals["tieu_de"]) if vals["tieu_de"].strip() else DEFAULT_TITLE_LEN
    desc_len = len(vals["mo_ta"]) if vals["mo_ta"].strip() else DEFAULT_DESC_LEN

    return pd.DataFrame([{
        "vehicle_age": vehicle_age,
        "so_km_da_di": so_km_da_di,
        "dung_tich_cc": dung_tich_cc,
        "title_len": title_len,
        "desc_len": desc_len,
        "km_missing_flag": km_missing_flag,
        "dung_tich_missing_flag": dung_tich_missing_flag,
        "pre_1980": pre_1980_flag,
        "thuong_hieu_grouped": vals["brand"],
        "dong_xe_grouped": vals["dong_xe"],
        "loai_xe": vals["loai_xe"],
        "xuat_xu_clean": vals["xuat_xu"],
        "quan": vals["quan"],
    }])


def build_feature_row_raw(brand, dong_xe, loai_xe, nam_dang_ky, so_km, dung_tich_cc,
                           xuat_xu, quan, tieu_de, mo_ta) -> pd.DataFrame:
    """Giong build_feature_row nhung nhan gia tri tho (vd tu file CSV upload) thay vi widget Streamlit."""
    nam = pd.to_numeric(nam_dang_ky, errors="coerce")
    if pd.isna(nam) or nam <= 1979:
        vehicle_age = REFERENCE_YEAR - 1979
        pre_1980_flag = 1
    else:
        vehicle_age = max(REFERENCE_YEAR - int(nam), 0)
        pre_1980_flag = 0

    so_km_num = pd.to_numeric(so_km, errors="coerce")
    if pd.isna(so_km_num):
        so_km_da_di = DEFAULT_KM
        km_missing_flag = 1
    else:
        so_km_da_di = float(so_km_num)
        km_missing_flag = 0

    dung_tich_num = pd.to_numeric(dung_tich_cc, errors="coerce")
    if pd.isna(dung_tich_num):
        dung_tich_cc_val = DEFAULT_DUNG_TICH
        dung_tich_missing_flag = 1
    else:
        dung_tich_cc_val = float(dung_tich_num)
        dung_tich_missing_flag = 0

    xuat_xu_val = xuat_xu if isinstance(xuat_xu, str) and xuat_xu.strip() in XUAT_XU_OPTIONS else "Không rõ"
    quan_val = quan if isinstance(quan, str) and quan.strip() else "Không rõ"
    brand_val = brand if isinstance(brand, str) and brand.strip() else "Khác"
    dong_xe_val = dong_xe if isinstance(dong_xe, str) and dong_xe.strip() else "Khác"
    loai_xe_val = loai_xe if isinstance(loai_xe, str) and loai_xe.strip() in LOAI_XE_OPTIONS else "Không rõ"
    tieu_de_val = tieu_de if isinstance(tieu_de, str) else ""
    mo_ta_val = mo_ta if isinstance(mo_ta, str) else ""
    title_len = len(tieu_de_val) if tieu_de_val.strip() else DEFAULT_TITLE_LEN
    desc_len = len(mo_ta_val) if mo_ta_val.strip() else DEFAULT_DESC_LEN

    return pd.DataFrame([{
        "vehicle_age": vehicle_age,
        "so_km_da_di": so_km_da_di,
        "dung_tich_cc": dung_tich_cc_val,
        "title_len": title_len,
        "desc_len": desc_len,
        "km_missing_flag": km_missing_flag,
        "dung_tich_missing_flag": dung_tich_missing_flag,
        "pre_1980": pre_1980_flag,
        "thuong_hieu_grouped": brand_val,
        "dong_xe_grouped": dong_xe_val,
        "loai_xe": loai_xe_val,
        "xuat_xu_clean": xuat_xu_val,
        "quan": quan_val,
    }])


def compute_anomaly_result(row: pd.DataFrame, gia_rao: float, brand: str, price_model, art: dict) -> dict:
    """Tinh diem bat thuong (3 tin hieu + composite) cho MOT xe, dung chung cho tab 2 va tab 3."""
    log_pred = price_model.predict(row)[0]
    gia_du_doan = np.expm1(log_pred)
    resid = gia_rao - gia_du_doan  # chi de hien thi (thang VND)
    # z-score tinh tren thang log(1+gia) de "qua re" va "qua dat" nhay nhu nhau
    resid_log = np.log1p(gia_rao) - log_pred

    resid_stats = art["resid_stats"]
    if brand in resid_stats.index:
        mean_r, std_r = resid_stats.loc[brand, "mean"], resid_stats.loc[brand, "std"]
    else:
        mean_r, std_r = art["resid_global_mean"], art["resid_global_std"]
    resid_z = (resid_log - mean_r) / std_r if std_r else 0.0

    quantiles = art["quantiles"]
    if brand in quantiles.index:
        p_low, p_high = quantiles.loc[brand, "p_low"], quantiles.loc[brand, "p_high"]
    else:
        p_low, p_high = art["global_p_low"], art["global_p_high"]
    below = max(p_low - gia_rao, 0) / p_low if p_low else 0.0
    above = max(gia_rao - p_high, 0) / p_high if p_high else 0.0
    percentile_violation = below + above

    unsup_row = [row.iloc[0][f] if f != "resid_z" else resid_z for f in art["unsup_features"]]
    X_new = art["imputer"].transform(pd.DataFrame([unsup_row], columns=art["unsup_features"]))
    X_new_scaled = art["scaler"].transform(X_new)

    iso_score_new = -art["iso"].score_samples(X_new_scaled)[0]
    lof_score_new = -art["lof"].score_samples(X_new_scaled)[0]
    ocsvm_score_new = -art["ocsvm"].decision_function(X_new_scaled)[0]
    # predict() cua tung mo hinh unsupervised: 1 = binh thuong, -1 = outlier (theo contamination da fit)
    unsup_votes = sum(p == -1 for p in [
        art["iso"].predict(X_new_scaled)[0], art["lof"].predict(X_new_scaled)[0], art["ocsvm"].predict(X_new_scaled)[0],
    ])

    iso_pct = percentile_rank(iso_score_new, art["iso_score_train"]) / 100
    lof_pct = percentile_rank(lof_score_new, art["lof_score_train"]) / 100
    ocsvm_pct = percentile_rank(ocsvm_score_new, art["ocsvm_score_train"]) / 100
    unsupervised_score = float(np.mean([iso_pct, lof_pct, ocsvm_pct]))

    resid_z_pct = percentile_rank(abs(resid_z), art["resid_z_abs_train"])
    percentile_violation_pct = percentile_rank(percentile_violation, art["percentile_violation_train"])
    unsupervised_pct = percentile_rank(unsupervised_score, art["unsupervised_score_train"])

    weights = art["weights"]
    composite = (
        weights["resid_z"] * resid_z_pct
        + weights["percentile_violation"] * percentile_violation_pct
        + weights["unsupervised_score"] * unsupervised_pct
    ) / sum(weights.values())

    is_anomaly = composite >= art["threshold"]
    if not is_anomaly:
        label = "Bình thường"
    elif resid < 0:
        label = "Nghi ngờ quá rẻ"
    else:
        label = "Nghi ngờ quá đắt"

    return {
        "gia_du_doan": gia_du_doan,
        "resid": resid,
        "resid_z": resid_z,
        "resid_z_pct": resid_z_pct,
        "resid_z_flag": bool(abs(resid_z) > 2),
        "percentile_violation": percentile_violation,
        "percentile_violation_pct": percentile_violation_pct,
        "percentile_violation_flag": bool(percentile_violation > 0),
        "iso_pct": iso_pct, "lof_pct": lof_pct, "ocsvm_pct": ocsvm_pct,
        "unsupervised_score": unsupervised_score,
        "unsupervised_pct": unsupervised_pct,
        "unsupervised_flag": bool(unsup_votes >= 2),
        "composite": composite,
        "is_anomaly": bool(is_anomaly),
        "label": label,
    }


st.set_page_config(page_title="Xe máy cũ — Dự đoán giá & Phát hiện bất thường", page_icon="🏍️", layout="centered")

st.title("🏍️ Đăng tin bán xe máy cũ")
st.caption(
    "Nhập thông tin xe như khi đăng tin trên Chợ Tốt. Tab 1 gợi ý mức giá bán hợp lý; "
    "Tab 2 kiểm tra 1 xe xem giá rao có bất thường không; Tab 3 kiểm tra hàng loạt nhiều xe cùng lúc từ file."
)

price_model, price_metrics = load_price_model()
anomaly_artifacts = load_anomaly_artifacts()

tab1, tab2, tab3 = st.tabs(["📈 Dự đoán giá", "🚨 Phát hiện bất thường", "📂 Kiểm tra hàng loạt"])

# ============================== TAB 1 — DU DOAN GIA ==============================
with tab1:
    with st.form("predict_form"):
        st.subheader("Thông tin xe")
        vals1 = render_vehicle_inputs("p")
        submitted1 = st.form_submit_button("💰 Dự đoán giá bán", use_container_width=True, type="primary")

    if submitted1:
        row = build_feature_row(vals1)
        price_pred = np.expm1(price_model.predict(row)[0])

        st.divider()
        st.subheader("Kết quả dự đoán")

        if price_metrics is not None:
            mae = price_metrics["mae_vnd"]
            low, high = max(price_pred - mae, 0), price_pred + mae
            st.metric("Giá bán gợi ý", format_vnd(price_pred))
            st.caption(f"Khoảng giá tham khảo: **{format_vnd(low)} — {format_vnd(high)}** "
                       f"(± sai số trung bình mô hình, R² ≈ {price_metrics['r2']:.2f})")
        else:
            st.metric("Giá bán gợi ý", format_vnd(price_pred))

        st.info(
            "⚠️ Đây là mức giá **tham khảo** từ mô hình học máy, không thay thế thẩm định "
            "thực tế (tình trạng máy móc, giấy tờ, thương lượng...).",
            icon="ℹ️",
        )

    st.divider()
    with st.expander("Về mô hình dự đoán giá"):
        st.write(
            "- Dữ liệu huấn luyện: hơn 7.000 tin đăng bán xe máy cũ trên Chợ Tốt (Tp.HCM, đến 07/2025).\n"
            "- Mô hình: Random Forest Regressor (đã tinh chỉnh siêu tham số), huấn luyện trên "
            "log(1+Giá) để giảm ảnh hưởng của các xe giá trị rất cao/thấp.\n"
            "- Các đặc trưng dùng để dự đoán: thương hiệu, dòng xe, loại xe, tuổi xe, số km đã đi, "
            "dung tích xi-lanh, xuất xứ, khu vực giao dịch, độ dài tiêu đề/mô tả."
        )
        if price_metrics is not None:
            st.write(f"- Đánh giá trên tập test: MAE ≈ {format_vnd(price_metrics['mae_vnd'])}, "
                     f"RMSE ≈ {format_vnd(price_metrics['rmse_vnd'])}, R² ≈ {price_metrics['r2']:.2f}.")

# ============================== TAB 2 — PHAT HIEN BAT THUONG ==============================
with tab2:
    if anomaly_artifacts is None:
        st.error(
            "Chưa tìm thấy file `models/anomaly_artifacts.joblib`. Hãy chạy "
            "`python build_anomaly_artifacts.py` trước để tạo các artifact cần thiết cho tab này."
        )
    else:
        with st.form("anomaly_form"):
            st.subheader("Thông tin xe")
            vals2 = render_vehicle_inputs("a")
            gia_rao = st.number_input(
                "Giá bạn muốn đăng bán (VND) *", min_value=500_000, max_value=2_000_000_000,
                value=30_000_000, step=500_000, key="a_gia_rao",
            )
            submitted2 = st.form_submit_button("🔎 Kiểm tra mức giá", use_container_width=True, type="primary")

        if submitted2:
            art = anomaly_artifacts
            row = build_feature_row(vals2)
            brand = vals2["brand"]
            result = compute_anomaly_result(row, gia_rao, brand, price_model, art)
            weights = art["weights"]

            if not result["is_anomaly"]:
                icon = "✅"
            elif result["label"] == "Nghi ngờ quá rẻ":
                icon = "🔵"
            else:
                icon = "🔴"

            st.divider()
            st.subheader("Kết quả kiểm tra")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Giá bạn rao", format_vnd(gia_rao))
            col_b.metric("Giá thị trường tham khảo", format_vnd(result["gia_du_doan"]))
            col_c.metric("Composite score", f"{result['composite']:.1f}/100")

            if not result["is_anomaly"]:
                st.success(f"{icon} **{result['label']}** — mức giá rao phù hợp với thị trường (ngưỡng cảnh báo top "
                           f"{art['top_k_percent']}%: {art['threshold']:.1f}/100).")
            else:
                st.warning(f"{icon} **{result['label'].upper()}** — điểm bất thường {result['composite']:.1f}/100 vượt "
                           f"ngưỡng cảnh báo top {art['top_k_percent']}% ({art['threshold']:.1f}/100).")

            with st.expander("Chi tiết 3 tín hiệu bất thường", expanded=True):
                breakdown = pd.DataFrame([
                    {
                        "Tín hiệu": "Residual-z theo thương hiệu",
                        "Giá trị thô": f"lệch {format_vnd(result['resid'])} (z={result['resid_z']:.2f})",
                        "Percentile rank": f"{result['resid_z_pct']:.1f}/100",
                        "Bất thường?": "Có" if result["resid_z_flag"] else "Không",
                        "Trọng số": f"{weights['resid_z']:.2f}",
                    },
                    {
                        "Tín hiệu": "Ngoài khoảng [P10, P90] theo thương hiệu",
                        "Giá trị thô": f"{result['percentile_violation']:.3f}",
                        "Percentile rank": f"{result['percentile_violation_pct']:.1f}/100",
                        "Bất thường?": "Có" if result["percentile_violation_flag"] else "Không",
                        "Trọng số": f"{weights['percentile_violation']:.2f}",
                    },
                    {
                        "Tín hiệu": "Unsupervised (Isolation Forest / LOF / One-Class SVM)",
                        "Giá trị thô": f"iso={result['iso_pct']:.2f}, lof={result['lof_pct']:.2f}, ocsvm={result['ocsvm_pct']:.2f}",
                        "Percentile rank": f"{result['unsupervised_pct']:.1f}/100",
                        "Bất thường?": "Có" if result["unsupervised_flag"] else "Không",
                        "Trọng số": f"{weights['unsupervised_score']:.2f}",
                    },
                ])
                st.dataframe(breakdown, hide_index=True, use_container_width=True)

            st.info(
                "⚠️ Điểm bất thường chỉ mang tính **tham khảo/sàng lọc**, không kết luận tin đăng là gian lận. "
                "Tín hiệu \"vi phạm khoảng giá min/max của Chợ Tốt\" trong nghiên cứu gốc không được dùng ở đây "
                "vì người đăng tin mới không có sẵn con số này.",
                icon="ℹ️",
            )

        st.divider()
        with st.expander("Về mô hình phát hiện bất thường"):
            st.write(
                "- Kết hợp 3 tín hiệu (điểm 0-100 mỗi tín hiệu), trọng số như nghiên cứu gốc "
                "(đã bỏ tín hiệu vi phạm khoảng giá Chợ Tốt, không khả dụng khi đăng tin mới):\n"
                f"  - Residual-z theo thương hiệu: {anomaly_artifacts['weights']['resid_z']:.2f}\n"
                f"  - Ngoài khoảng [P10, P90] theo thương hiệu: {anomaly_artifacts['weights']['percentile_violation']:.2f}\n"
                f"  - Unsupervised (Isolation Forest/LOF/One-Class SVM): {anomaly_artifacts['weights']['unsupervised_score']:.2f}\n"
                f"- Ngưỡng cảnh báo: top {anomaly_artifacts['top_k_percent']}% điểm composite cao nhất trên tập huấn luyện "
                f"({anomaly_artifacts['threshold']:.1f}/100)."
            )

# ============================== TAB 3 — KIEM TRA HANG LOAT ==============================
with tab3:
    if anomaly_artifacts is None:
        st.error(
            "Chưa tìm thấy file `models/anomaly_artifacts.joblib`. Hãy chạy "
            "`python build_anomaly_artifacts.py` trước để tạo các artifact cần thiết cho tab này."
        )
    else:
        st.subheader("Tải lên danh sách nhiều xe")
        st.write(
            "Tải lên file CSV chứa nhiều xe — hệ thống sẽ dự đoán giá thị trường và chấm điểm bất thường "
            "cho **từng xe** (theo từng tín hiệu và điểm tổng hợp), không cần nhập tay từng chiếc."
        )

        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            if os.path.exists(SAMPLE_BATCH_PATH):
                with open(SAMPLE_BATCH_PATH, "rb") as f:
                    st.download_button(
                        "⬇️ Tải file mẫu để test", data=f.read(),
                        file_name="sample_batch_upload.csv", mime="text/csv",
                        use_container_width=True,
                    )
        with col_info:
            st.caption(
                "File mẫu có sẵn 12 xe (gồm vài ca cố tình quá rẻ/quá đắt, thiếu số km/dung tích, "
                "hãng lạ, xe cổ trước 1980) để bạn thử ngay tính năng."
            )

        with st.expander("Định dạng file CSV cần có"):
            st.write("Cột **bắt buộc**: `thuong_hieu`, `dong_xe`, `loai_xe`, `nam_dang_ky`, `gia_rao`")
            st.write("Cột tuỳ chọn (bỏ trống nếu không có): `so_km`, `dung_tich_cc`, `xuat_xu`, `quan`, `tieu_de`, `mo_ta`")
            st.write(f"`loai_xe` phải là một trong: {', '.join(LOAI_XE_OPTIONS)}")

        uploaded = st.file_uploader("Chọn file CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded, encoding="utf-8-sig")
            except Exception as exc:
                st.error(f"Không đọc được file CSV: {exc}")
                batch_df = None

            if batch_df is not None:
                missing_cols = [c for c in BATCH_REQUIRED_COLUMNS if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"File thiếu cột bắt buộc: {', '.join(missing_cols)}")
                else:
                    for c in BATCH_OPTIONAL_COLUMNS:
                        if c not in batch_df.columns:
                            batch_df[c] = np.nan

                    art = anomaly_artifacts
                    results = []
                    progress = st.progress(0.0, text="Đang xử lý...")
                    n = len(batch_df)
                    for i, r in batch_df.iterrows():
                        gia_rao_i = pd.to_numeric(r.get("gia_rao"), errors="coerce")
                        if pd.isna(gia_rao_i):
                            results.append({"__error__": "Thiếu giá rao (gia_rao)"})
                            progress.progress((i + 1) / n)
                            continue
                        brand_i = r.get("thuong_hieu") if isinstance(r.get("thuong_hieu"), str) and r.get("thuong_hieu").strip() else "Khác"
                        row_i = build_feature_row_raw(
                            brand=r.get("thuong_hieu"), dong_xe=r.get("dong_xe"), loai_xe=r.get("loai_xe"),
                            nam_dang_ky=r.get("nam_dang_ky"), so_km=r.get("so_km"), dung_tich_cc=r.get("dung_tich_cc"),
                            xuat_xu=r.get("xuat_xu"), quan=r.get("quan"), tieu_de=r.get("tieu_de"), mo_ta=r.get("mo_ta"),
                        )
                        res = compute_anomaly_result(row_i, float(gia_rao_i), brand_i, price_model, art)
                        res["__error__"] = None
                        results.append(res)
                        progress.progress((i + 1) / n)
                    progress.empty()

                    out = batch_df.copy()
                    out["Giá thị trường tham khảo"] = [r.get("gia_du_doan") for r in results]
                    out["Residual-z bất thường?"] = ["Có" if r.get("resid_z_flag") else ("Không" if r.get("resid_z_flag") is not None else "") for r in results]
                    out["Ngoài P10-P90 bất thường?"] = ["Có" if r.get("percentile_violation_flag") else ("Không" if r.get("percentile_violation_flag") is not None else "") for r in results]
                    out["Unsupervised bất thường?"] = ["Có" if r.get("unsupervised_flag") else ("Không" if r.get("unsupervised_flag") is not None else "") for r in results]
                    out["Composite score"] = [round(r.get("composite"), 1) if r.get("composite") is not None else None for r in results]
                    out["Kết luận tổng hợp"] = [r.get("label") if r.get("label") is not None else r.get("__error__") for r in results]

                    n_anomaly = sum(1 for r in results if r.get("is_anomaly"))
                    n_error = sum(1 for r in results if r.get("__error__"))

                    st.divider()
                    st.subheader("Kết quả kiểm tra hàng loạt")
                    col_x, col_y, col_z = st.columns(3)
                    col_x.metric("Tổng số xe", n)
                    col_y.metric("Bị gắn cờ bất thường", n_anomaly)
                    col_z.metric("Lỗi dữ liệu", n_error)

                    st.dataframe(out, hide_index=True, use_container_width=True)

                    csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Tải kết quả (CSV)", data=csv_bytes,
                        file_name="ket_qua_kiem_tra_hang_loat.csv", mime="text/csv",
                    )

                    st.info(
                        "⚠️ Cột \"Bất thường?\" của từng tín hiệu là cờ tham khảo riêng lẻ "
                        "(residual-z theo ngưỡng |z|>2; ngoài khoảng P10-P90; unsupervised theo đa số phiếu "
                        "của Isolation Forest/LOF/One-Class SVM). \"Kết luận tổng hợp\" dựa trên composite "
                        f"score kết hợp cả 3 (ngưỡng top {anomaly_artifacts['top_k_percent']}%).",
                        icon="ℹ️",
                    )
