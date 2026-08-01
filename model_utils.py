"""
model_utils.py
================
Load bundle model XGBoost (xgb_price_model.pkl) và xây dựng vector đặc trưng
(feature vector) đúng định dạng mà model đã được huấn luyện, để gọi .predict().

QUAN TRỌNG - CÁC GIẢ ĐỊNH (assumptions):
File .pkl chỉ chứa model + một số metadata (feature_cols, onehot_cols,
dung_tich_order, dung_tich_median, reference_year, km_median_global,
hang_group_valid, dong_group_valid) — KHÔNG chứa hàm tiền xử lý gốc.
Mình đã suy ra công thức hợp lý nhất cho từng cột dựa trên tên cột, nhưng
3 chỗ dưới đây là SUY LUẬN, bạn nên đối chiếu lại với notebook huấn luyện gốc:

  1. `Nam_khong_ro`: mình map từ checkbox "Xe đăng ký trước năm 1980" (vì UI
     gốc không có ô "không rõ năm đăng ký" riêng). Nếu notebook gốc dùng
     logic khác, sửa lại trong `build_features()`.
  2. `Km_lam_tron`: mình làm tròn Km về bội số của 1000
     (round(km/1000)*1000). Đây là suy đoán hợp lý nhất từ tên cột.
  3. `Phan_khuc_bin`: không có cách nào suy ra chính xác 100% chỉ từ tên
     cột — mình tạm dùng `Dung_tich_ord` làm phân khúc (0-3). Nếu bạn có
     công thức gốc, thay hàm `_phan_khuc_bin()` bên dưới.

Nếu bạn có sẵn hàm tiền xử lý gốc (vd. `build_features_from_notebook`),
đưa cho mình để thay thế cho chắc chắn 100% khớp với lúc train.
"""

import pickle
import numpy as np
import pandas as pd


def load_model_bundle(path: str):
    """Load toàn bộ dict (model + metadata) từ file .pkl."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def get_options(bundle: dict) -> dict:
    """
    Suy ra danh sách lựa chọn hợp lệ cho từng trường one-hot TRỰC TIẾP từ
    feature_cols của model (không hardcode) — nên luôn khớp với model,
    kể cả khi bạn train lại và đổi danh sách hãng/dòng xe.
    """
    feature_cols = bundle["feature_cols"]
    onehot_cols = bundle["onehot_cols"]  # vd: ['Loại xe','Xuat_xu_clean','Hang_group','Dong_group']

    options = {}
    for oc in onehot_cols:
        prefix = f"{oc}_"
        opts = [c[len(prefix):] for c in feature_cols if c.startswith(prefix)]
        options[oc] = sorted(opts)

    # Danh mục hãng / dòng "hợp lệ" đầy đủ (bao gồm cả nhóm gộp vào baseline)
    options["hang_group_valid"] = sorted(
        {str(x) for x in bundle["hang_group_valid"] if isinstance(x, str)}
    )
    options["dong_group_valid"] = sorted(
        {str(x) for x in bundle["dong_group_valid"] if isinstance(x, str)}
    )

    # Dung tích: sắp theo thứ tự ordinal đã lưu trong model
    dung_tich_order = bundle["dung_tich_order"]
    options["dung_tich_labels"] = sorted(dung_tich_order, key=lambda k: dung_tich_order[k])

    return options


# ============================================================
# BẢNG ÁNH XẠ HÃNG -> DÒNG XE (để dropdown Dòng xe phụ thuộc Hãng xe)
# ============================================================
# GIẢ ĐỊNH: file .pkl không lưu quan hệ hãng-dòng (chỉ lưu 2 danh sách
# phẳng riêng biệt), nên mình tự đối chiếu bằng kiến thức thực tế thị
# trường xe máy VN. Nếu có dòng nào gán sai hãng, sửa trực tiếp ở đây.
HANG_DONG_MAP = {
    "Honda": [
        "Wave", "Dream", "Future", "Cub", "Blade", "Air Blade", "Vision",
        "Lead", "SH", "SH Mode", "Winner", "Winner X", "CB", "CBR",
        "MSX 125", "@", "PCX", "Click", "Dylan", "Spacy", "Chaly",
        "Sonic", "Win", "PS",
    ],
    "Yamaha": [
        "Sirius", "Exciter", "Jupiter", "Nouvo", "Grande", "Janus",
        "Nvx", "FZ", "Luvias", "Mio", "Hayate", "Nozza",
    ],
    "Suzuki": ["Raider", "Satria", "GSX", "R", "Shark"],
    "Piaggio": ["Vespa", "LX", "Sprint", "GTS", "Liberty", "PG-1"],
    "SYM": ["Attila", "Elegant", "Elizabeth", "Sport / Xipo"],
    "Kymco": [],
    "Kawasaki": [],
    "Ducati": [],
    "Detech": ["67"],
}

# Danh sách hãng gắn với từng phân khúc, dùng để lọc dropdown Hãng xe
# theo tab phân khúc người dùng chọn (giống mockup: Phổ thông / Cổ điển / PKL)
SEGMENT_HANG = {
    "Xe phổ thông": ["Honda", "Yamaha", "Suzuki", "SYM", "Kymco", "Khác"],
    "Xe cổ điển": ["Detech", "Honda"],  # 67, Cub, Dream, Win, Chaly (dòng cổ)
    "Phân khối lớn": ["Ducati", "Kawasaki", "Honda", "Suzuki", "Piaggio"],
}
SEGMENT_DUNG_TICH_GOI_Y = {
    "Xe phổ thông": "50 - 100 cc",
    "Xe cổ điển": "50 - 100 cc",
    "Phân khối lớn": "Trên 175 cc",
}


def get_dong_options(bundle: dict, hang: str, opts: dict = None) -> list:
    """Trả về danh sách Dòng xe hợp lệ cho 1 Hãng xe cụ thể (dropdown phụ thuộc)."""
    if opts is None:
        opts = get_options(bundle)
    all_dong = set(opts["dong_group_valid"])
    mapped = [d for d in HANG_DONG_MAP.get(hang, []) if d in all_dong]
    if not mapped:
        # Hãng chưa có trong bảng ánh xạ thủ công -> hiện toàn bộ danh sách
        # dòng xe để không chặn người dùng, kèm "Khác" ở cuối.
        mapped = sorted(all_dong)
    if "Khác" in all_dong and "Khác" not in mapped:
        mapped = mapped + ["Khác"]
    return mapped


def get_hang_options_for_segment(bundle: dict, segment: str, opts: dict = None) -> list:
    """Lọc danh sách Hãng xe theo phân khúc (phổ thông/cổ điển/PKL)."""
    if opts is None:
        opts = get_options(bundle)
    all_hang = set(opts["hang_group_valid"])
    segment_list = [h for h in SEGMENT_HANG.get(segment, []) if h in all_hang]
    if not segment_list:
        segment_list = sorted(all_hang)
    return segment_list


def _dung_tich_ord(bundle, dung_tich_label, dung_tich_unknown):
    order_map = bundle["dung_tich_order"]
    if dung_tich_unknown or dung_tich_label not in order_map:
        return bundle["dung_tich_median"], 1
    return float(order_map[dung_tich_label]), 0


def _phan_khuc_bin(dung_tich_ord):
    # GIẢ ĐỊNH: xem docstring đầu file — thay bằng công thức gốc nếu có.
    return int(min(max(round(dung_tich_ord), 0), 3))


def build_features(
    bundle,
    *,
    hang: str,
    dong: str,
    loai_xe: str,
    xuat_xu: str,
    dung_tich_label: str,
    dung_tich_unknown: bool,
    reg_before_1980: bool,
    nam_dang_ky: int,
    km: float,
    km_unknown: bool,
) -> pd.DataFrame:
    """Xây dựng 1 dòng DataFrame đúng thứ tự cột `feature_cols` của model."""

    feature_cols = bundle["feature_cols"]
    reference_year = bundle["reference_year"]
    km_median_global = bundle["km_median_global"]
    hang_group_valid = set(bundle["hang_group_valid"])
    dong_group_valid = set(bundle["dong_group_valid"])

    row = {col: 0 for col in feature_cols}

    # --- Tuổi xe ---
    nam_khong_ro = 1 if reg_before_1980 else 0
    tuoi_xe = (reference_year - 1980) if reg_before_1980 else max(reference_year - int(nam_dang_ky), 0)
    row["Tuoi_xe_final"] = tuoi_xe
    row["Nam_khong_ro"] = nam_khong_ro

    # --- Số Km ---
    km_final = km_median_global if km_unknown else float(km)
    row["Km_final"] = km_final
    row["Km_lam_tron"] = round(km_final / 1000) * 1000

    # --- Dung tích ---
    dung_tich_ord, dung_tich_kr = _dung_tich_ord(bundle, dung_tich_label, dung_tich_unknown)
    row["Dung_tich_ord"] = dung_tich_ord
    row["Dung_tich_khong_ro"] = dung_tich_kr
    row["Phan_khuc_bin"] = _phan_khuc_bin(dung_tich_ord)

    # --- One-hot: Loại xe ---
    col = f"Loại xe_{loai_xe}"
    if col in row:
        row[col] = 1

    # --- One-hot: Xuất xứ ---
    col = f"Xuat_xu_clean_{xuat_xu}"
    if col in row:
        row[col] = 1
    else:
        col = "Xuat_xu_clean_Không rõ"
        if col in row:
            row[col] = 1

    # --- One-hot: Hãng xe (fallback -> Khác nếu hãng lạ) ---
    hang_final = hang if hang in hang_group_valid else "Khác"
    col = f"Hang_group_{hang_final}"
    if col in row:
        row[col] = 1

    # --- One-hot: Dòng xe (fallback -> Khác nếu dòng lạ) ---
    dong_final = dong if dong in dong_group_valid else "Khác"
    col = f"Dong_group_{dong_final}"
    if col in row:
        row[col] = 1

    return pd.DataFrame([row], columns=feature_cols)


def predict_price(bundle, features_df: pd.DataFrame) -> float:
    """
    Dự đoán giá xe (VNĐ).

    QUAN TRỌNG: model trả về giá trị ở thang LOG (vd. raw prediction ~17
    cho 1 xe giá ~24 triệu, vì e^17 ≈ 24.000.000). Đây gần như chắc chắn
    do lúc train, biến mục tiêu là log(giá) hoặc log1p(giá) — cách làm rất
    phổ biến với dữ liệu giá có phân phối lệch phải. Mình dùng expm1() để
    biến đổi ngược lại VNĐ thực tế.

    Nếu sau khi sửa mà giá dự đoán vẫn sai lệch nhiều so với giá thị trường
    thực tế, khả năng cao notebook gốc dùng log1p (không phải log tự nhiên
    thường), hoặc có thêm bước chuẩn hoá khác (StandardScaler, chia cho
    1 triệu...) — lúc đó cần đối chiếu lại notebook train để chỉnh đúng.
    """
    model = bundle["model"]
    raw_pred = float(np.asarray(model.predict(features_df)).ravel()[0])
    gia_vnd = np.expm1(raw_pred)
    return float(gia_vnd)


def classify_price(gia_rao: float, gia_du_doan: float, threshold: float = 0.15):
    """So sánh giá rao với giá dự đoán, trả về (nhãn, % chênh lệch)."""
    if gia_du_doan <= 0:
        return "Không xác định", 0.0
    chenh_lech = (gia_rao - gia_du_doan) / gia_du_doan
    if chenh_lech > threshold:
        nhan = "Quá đắt"
    elif chenh_lech < -threshold:
        nhan = "Quá rẻ"
    else:
        nhan = "Hợp lý"
    return nhan, chenh_lech * 100