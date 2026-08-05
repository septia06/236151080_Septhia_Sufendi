import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# =========================================================
# CONFIG
# =========================================================
# NILAI AWAL/DEFAULT — akan dihitung ulang pada bagian Penentuan Jumlah Cluster.
# Kurva Elbow/WCSS ditampilkan pada K=2–9. Sesuai arahan bimbingan, kandidat
# keputusan adalah K=2, K=3, dan K=4; distribusi diperiksa sebelum Silhouette.
best_k = 3
elbow_k = 3
RANDOM_STATE = 42
APP_VERSION = "Final Dospem Lengkap — Asal n, mean, min, max, centroid, dan K final"

st.set_page_config(page_title="Clustering Harga Gula (K-Means)", page_icon="🍬", layout="wide")

# =========================================================
# HELPERS - PARSING & CLEANING
# =========================================================
def to_numeric_safe(s: pd.Series) -> pd.Series:
    """
    Mengubah kolom Harga menjadi numerik dengan aman.

    PENTING: jika nilai pada Excel SUDAH berupa angka asli (int/float, misalnya
    18000.0), kita TIDAK BOLEH mengubahnya ke teks lalu menghapus titik -- sebab
    titik pada "18000.0" adalah titik desimal Python, bukan pemisah ribuan gaya
    Indonesia. Kalau tetap dihapus, "18000.0" -> "180000" (salah dikali 10).
    String-cleaning (hapus "Rp", hapus titik ribuan, ubah koma jadi desimal)
    HANYA dilakukan untuk nilai yang memang berupa teks berformat, misalnya
    "Rp 18.000" atau "18.000,50".
    """
    # Jalur cepat: kolom sudah numerik murni (int/float) -> pakai apa adanya,
    # tanpa manipulasi string sama sekali.
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    def _parse_satu_nilai(v):
        if pd.isna(v):
            return np.nan
        # Nilai per-baris yang kebetulan sudah numerik (mis. kolom object berisi
        # campuran angka & teks) -> pakai langsung, jangan lewat string-cleaning.
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)

        txt = str(v).strip()
        txt = txt.replace("Rp", "").replace(" ", "")
        if not txt:
            return np.nan

        # Format Indonesia umum: "18.000" (titik = ribuan) atau "18.000,50"
        # (titik = ribuan, koma = desimal). Kita hapus titik sebagai pemisah
        # ribuan HANYA pada string asli, lalu ubah koma jadi titik desimal.
        txt = txt.replace(".", "")
        txt = txt.replace(",", ".")
        return txt

    parsed = s.map(_parse_satu_nilai)
    return pd.to_numeric(parsed, errors="coerce")

def normalize_header_name(h: str) -> str:
    x = str(h).strip()
    xl = x.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    if xl in ["tangga", "tgl", "tanggal"]:
        return "Tanggal"
    if xl == "no":
        return "No"
    if xl in [
        "jenis", "jenis_komoditi", "jenis_komiditi",
        "jenis_komoditi_", "jenis_komiditi_"
    ]:
        return "Jenis_Komoditi"
    if xl in ["het_ha", "het", "ha"]:
        return "HET_HA"
    return x

def smart_split_header(s: str) -> list:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if "\t" in s:
        return [p.strip() for p in s.split("\t") if p.strip()]
    import re
    return [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()]

BULAN_ID = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4,
    "mei": 5, "juni": 6, "juli": 7, "agustus": 8,
    "september": 9, "oktober": 10, "november": 11, "desember": 12,
}

def parse_tanggal_indonesia(value):
    """Membaca tanggal dari Excel mitra, termasuk format '7 Maret 2025'."""
    if pd.isna(value):
        return pd.NaT

    dt = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if not pd.isna(dt):
        return dt

    import re
    s = str(value).strip().lower()
    m = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", s)
    if m and m.group(2) in BULAN_ID:
        return pd.Timestamp(year=int(m.group(3)), month=BULAN_ID[m.group(2)], day=int(m.group(1)))

    return pd.NaT

def is_sheet_harian_mitra(sheet_name: str) -> bool:
    """Sheet harian mitra berbentuk '2 Januari 2025'. Sheet non-harian seperti 'Rata-Rata' dan 'Lamin Etam' tidak dipakai."""
    import re
    bulan = "|".join([b.capitalize() for b in BULAN_ID.keys()])
    return bool(re.match(rf"^\s*\d{{1,2}}\s+({bulan})\s+\d{{4}}\s*$", str(sheet_name), flags=re.IGNORECASE))

def build_data_mentah_semua_komoditi_dari_mitra(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    Mengubah file mitra yang berisi banyak sheet harian menjadi satu tabel Data Mentah RAW.
    PENTING: fungsi ini SENGAJA TIDAK memfilter komoditi apa pun -- semua komoditi yang
    tercatat pada tiap sheet harian (beras, minyak goreng, gula pasir, dll) ikut disertakan.
    Ini murni tahap INPUT/PENGGABUNGAN data mentah. Pemilihan khusus komoditi Gula Pasir
    dilakukan belakangan sebagai bagian dari tahap Data Cleaning, bukan di sini.
    Output: No, Tanggal, Jenis_Komoditi, HET_HA, lalu kolom kabupaten/kota
    (bisa lebih dari satu baris per tanggal -- satu baris untuk setiap komoditi pada sheet itu).
    """
    rows = []
    daily_sheets = [s for s in xl.sheet_names if is_sheet_harian_mitra(s)]

    for sheet in daily_sheets:
        try:
            raw_sheet = xl.parse(sheet, header=None, )
        except Exception:
            continue

        if raw_sheet.shape[0] < 5:
            continue

        tanggal = parse_tanggal_indonesia(raw_sheet.iloc[2, 0])
        if pd.isna(tanggal):
            tanggal = parse_tanggal_indonesia(sheet)
        if pd.isna(tanggal):
            continue

        header = raw_sheet.iloc[3].tolist()
        header_norm = [normalize_header_name(h) for h in header]

        try:
            jenis_idx = header_norm.index("Jenis_Komoditi")
        except ValueError:
            jenis_candidates = [i for i, h in enumerate(header) if "jenis" in str(h).lower()]
            if not jenis_candidates:
                continue
            jenis_idx = jenis_candidates[0]

        try:
            het_idx = header_norm.index("HET_HA")
        except ValueError:
            het_candidates = [i for i, h in enumerate(header) if "het" in str(h).lower()]
            het_idx = het_candidates[0] if het_candidates else jenis_idx + 1

        kab_indices = [
            i for i in range(het_idx + 1, len(header))
            if not pd.isna(header[i]) and str(header[i]).strip()
        ]

        if not kab_indices:
            continue

        # Ambil SEMUA baris komoditi di bawah header (tidak disaring ke gula pasir saja).
        data_rows = raw_sheet.iloc[4:]
        for _, komoditi_row in data_rows.iterrows():
            jenis_val = komoditi_row.iloc[jenis_idx] if jenis_idx < len(komoditi_row) else np.nan
            if pd.isna(jenis_val) or not str(jenis_val).strip():
                continue

            row = {
                "Tanggal": tanggal,
                "Jenis_Komoditi": str(jenis_val).strip(),
                "HET_HA": komoditi_row.iloc[het_idx] if het_idx < len(komoditi_row) else np.nan,
            }

            for i in kab_indices:
                kab_name = str(header[i]).strip()
                row[kab_name] = komoditi_row.iloc[i] if i < len(komoditi_row) else np.nan

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["Tanggal", "Jenis_Komoditi"]).drop_duplicates(
        subset=["Tanggal", "Jenis_Komoditi"], keep="last"
    ).reset_index(drop=True)
    df.insert(0, "No", range(1, len(df) + 1))

    # Urutan kolom dibuat tetap agar tabel mudah dibaca dan konsisten dengan format mentahan.
    base_cols = ["No", "Tanggal", "Jenis_Komoditi", "HET_HA"]
    kab_cols = [c for c in df.columns if c not in base_cols]
    return df[base_cols + kab_cols]

# =========================================================
# CACHING — PENTING UNTUK PERFORMA
# =========================================================
# Streamlit menjalankan ULANG SELURUH SCRIPT dari atas setiap kali ada interaksi
# apa pun (klik expander, centang checkbox, dsb). Tanpa cache, ini berarti SEMUA
# sheet harian di file Excel mitra (bisa ratusan sheet untuk 1 tahun data) akan
# di-parse ULANG dari awal pada SETIAP interaksi -- inilah sumber utama aplikasi
# terasa lambat. Fungsi-fungsi di bawah ini di-cache berdasarkan ISI file (bytes),
# sehingga proses parsing Excel hanya dijalankan SEKALI selama file belum berubah.
#
# PENTING (perbaikan performa): sebelumnya pd.ExcelFile(...) dibuat ULANG di
# beberapa tempat berbeda (saat ambil daftar sheet, saat parse semua sheet
# harian, saat build data mentah mitra, dan saat baca 1 sheet non-mitra).
# Untuk file dengan ratusan sheet harian, setiap pd.ExcelFile(...) berarti
# membuka & mem-parsing ULANG seluruh struktur workbook dari openpyxl -- kalau
# dibuka 3-4 kali, biaya loadingnya jadi 3-4x lipat. get_excel_file() di bawah
# ini di-cache dengan st.cache_resource supaya workbook HANYA dibuka SEKALI,
# lalu objek ExcelFile yang sama dipakai ulang di semua fungsi lain.
@st.cache_resource(show_spinner="Membuka file Excel...")
def get_excel_file(file_bytes: bytes) -> pd.ExcelFile:
    return pd.ExcelFile(io.BytesIO(file_bytes))

@st.cache_data(show_spinner="Membaca seluruh sheet harian dari Excel...")
def cached_parse_all_daily_sheets(file_bytes: bytes, sheet_names: tuple) -> pd.DataFrame:
    xl_local = get_excel_file(file_bytes)
    parts = []
    for sheet in sheet_names:
        try:
            sheet_df = xl_local.parse(sheet, header=None)
        except Exception:
            continue
        sheet_df = sheet_df.copy()
        sheet_df.insert(0, "Sheet", sheet)
        parts.append(sheet_df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

@st.cache_data(show_spinner="Menggabungkan data mentah dari seluruh sheet harian...")
def cached_build_data_mentah_mitra(file_bytes: bytes, sheet_names: tuple) -> pd.DataFrame:
    xl_local = get_excel_file(file_bytes)
    return build_data_mentah_semua_komoditi_dari_mitra(xl_local)

@st.cache_data(show_spinner="Membaca sheet Excel...")
def cached_read_excel_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    xl_local = get_excel_file(file_bytes)
    return xl_local.parse(sheet_name, header=None)

def find_header_row(raw: pd.DataFrame) -> int:
    for i in range(min(60, len(raw))):
        row0 = str(raw.iloc[i, 0]).strip().lower()

        if ("no|" in row0) and (("tanggal" in row0) or ("tangga" in row0) or ("tgl" in row0)) and ("jenis" in row0):
            return i

        row_vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if ("no" in row_vals) and ("tanggal" in row_vals):
            return i

    return 0

def build_table_from_raw(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    header_first = str(raw.iloc[header_row, 0]).strip()
    data = raw.iloc[header_row + 1:].copy().reset_index(drop=True)

    left_headers = smart_split_header(header_first) if "|" in header_first else ([header_first] if header_first else [])
    left_headers = [normalize_header_name(h) for h in left_headers]

    header_rest = raw.iloc[header_row, 1:].tolist()
    tokens = []
    for cell in header_rest:
        if pd.isna(cell):
            continue
        cell_str = str(cell).strip()
        if not cell_str:
            continue
        spl = smart_split_header(cell_str)
        tokens.extend(spl if len(spl) > 1 else [cell_str])

    tokens = [normalize_header_name(t) for t in tokens if str(t).strip()]
    right_headers = tokens if len(tokens) else [f"Col_{j}" for j in range(1, raw.shape[1])]

    if len(left_headers) >= 2 and ("Tanggal" in left_headers):
        left_split = data.iloc[:, 0].astype(str).str.split("|", expand=True)

        if left_split.shape[1] < len(left_headers):
            for _ in range(len(left_headers) - left_split.shape[1]):
                left_split[left_split.shape[1]] = np.nan

        left_split = left_split.iloc[:, :len(left_headers)]
        left_split.columns = left_headers

        available_right_cols = min(len(right_headers), max(0, data.shape[1] - 1))
        right_part = data.iloc[:, 1:1 + available_right_cols].copy()
        right_part.columns = right_headers[:available_right_cols]

        df = pd.concat([left_split, right_part], axis=1)

    else:
        headers = raw.iloc[header_row].tolist()
        headers = [normalize_header_name(h) if pd.notna(h) else "" for h in headers]
        df = raw.iloc[header_row + 1:].copy()
        df.columns = headers
        df = df.reset_index(drop=True)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c not in ["", "None"]]]
    return df

def iqr_clip(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return series, 0, np.nan, np.nan

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    cnt = int(((series < low) | (series > high)).sum())
    return series.clip(low, high), cnt, low, high

def add_period(df: pd.DataFrame, date_col: str, mode: str) -> pd.Series:
    d = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if mode == "Harian":
        return d.dt.to_period("D").astype(str)
    if mode == "Mingguan":
        return d.dt.to_period("W").astype(str)
    return d.dt.to_period("M").astype(str)

def df_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def make_trend_xdate(trend: pd.DataFrame, period_mode: str) -> pd.DataFrame:
    trend = trend.copy()

    if period_mode == "Harian":
        trend["x_date"] = pd.to_datetime(trend["Periode"], errors="coerce")
    elif period_mode == "Mingguan":
        start_week = trend["Periode"].astype(str).str.split(r"[\/\s]", n=1, expand=True)[0]
        trend["x_date"] = pd.to_datetime(start_week, errors="coerce")
    else:
        trend["x_date"] = pd.to_datetime(trend["Periode"] + "-01", errors="coerce")

    trend = trend.dropna(subset=["x_date"]).sort_values("x_date")
    return trend

def apply_nice_xaxis(ax, period_mode: str):
    if period_mode == "Harian":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    elif period_mode == "Mingguan":
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# =========================================================
# HELPERS - FITUR
# =========================================================
FEATURE_DEFINITIONS = [
    {"Fitur": "Harga_Rata2", "Formula": "mean(Harga)", "Peran": "Fitur clustering",
     "Deskripsi": "Rata-rata harga gula pasir tiap kabupaten/kota selama periode pengamatan."},
    {"Fitur": "Harga_Min", "Formula": "min(Harga)", "Peran": "Fitur clustering",
     "Deskripsi": "Harga terendah yang pernah tercatat pada kabupaten/kota tersebut."},
    {"Fitur": "Harga_Max", "Formula": "max(Harga)", "Peran": "Fitur clustering",
     "Deskripsi": "Harga tertinggi yang pernah tercatat pada kabupaten/kota tersebut."},
    {"Fitur": "Harga_Std", "Formula": "std(Harga)", "Peran": "Fitur clustering",
     "Deskripsi": "Standar deviasi harga; mengukur seberapa besar fluktuasi/naik-turun harga pada kabupaten/kota tersebut."},
    {"Fitur": "Harga_Skewness", "Formula": "skew(Harga)", "Peran": "Fitur clustering",
     "Deskripsi": "Kemiringan (skewness) distribusi harga; mengukur apakah harga cenderung condong ke nilai tinggi atau rendah dibanding rata-ratanya."},
]

# Mengikuti pedoman bimbingan: K-Means memakai lima fitur harga secara langsung
# (Harga rata-rata, Harga minimum, Harga maksimum, Standar deviasi, Skewness/kemiringan),
# bukan skor komposit turunan. Std & Skewness ditambahkan agar fitur juga menangkap
# fluktuasi dan bentuk sebaran harga, bukan hanya titik rata-rata/min/max.
FEATURE_COLS = ["Harga_Rata2", "Harga_Min", "Harga_Max", "Harga_Std", "Harga_Skewness"]
DESCRIPTIVE_COLS = ["Tanggal_Awal", "Tanggal_Akhir", "Jumlah_Data_Harian"]

def posisi_relatif(value, semua_nilai, label_tinggi="tinggi", label_rendah="rendah", label_sedang="sedang"):
    """Menentukan posisi relatif sebuah nilai dibanding kumpulan nilai (cluster) lain."""
    semua_nilai = list(semua_nilai)
    if len(semua_nilai) <= 1:
        return label_sedang
    n_lebih_kecil_sama = sum(1 for v in semua_nilai if v <= value)
    pct_rank = (n_lebih_kecil_sama - 1) / max(1, (len(semua_nilai) - 1))
    if pct_rank >= 0.66:
        return label_tinggi
    elif pct_rank <= 0.33:
        return label_rendah
    return label_sedang

# =========================================================
# HELPERS - K-MEANS & EVALUASI
# =========================================================
def _jarak_garis_elbow(ks, inertias):
    """Skor elbow berbasis jarak titik WCSS ke garis awal-akhir (kneedle sederhana)."""
    if len(ks) < 3:
        return int(ks[0]), {int(ks[0]): 0.0}

    x = np.array(ks, dtype=float)
    y = np.array(inertias, dtype=float)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    denom = np.linalg.norm(p2 - p1)
    if denom == 0:
        return int(ks[0]), {int(k): 0.0 for k in ks}

    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        dist = np.abs(np.cross(p2 - p1, p1 - p)) / denom
        distances.append(float(dist))
    skor = {int(k): float(d) for k, d in zip(ks, distances)}
    return int(ks[int(np.argmax(distances))]), skor


def _siku_perlambatan_wcss(ks, inertias):
    """Mencari siku praktis dari perubahan penurunan WCSS.

    Titik siku praktis dibaca pada K ketika penurunan WCSS dari K sebelumnya masih besar,
    tetapi penurunan setelah K tersebut mulai mengecil. Untuk data kecil seperti jumlah
    kabupaten/kota, pembacaan ini biasanya lebih aman daripada langsung memakai titik
    jarak-terjauh yang sering tertarik ke K tinggi ketika K maksimum mendekati jumlah data.
    """
    if len(ks) < 3:
        return int(ks[0]), {int(ks[0]): 0.0}

    ks_arr = np.array(ks, dtype=int)
    y = np.array(inertias, dtype=float)
    scores = {}

    # Kandidat siku ada di tengah: K_i, dengan penurunan sebelum dan sesudahnya.
    for i in range(1, len(ks_arr) - 1):
        prev_drop = float(y[i - 1] - y[i])
        next_drop = float(y[i] - y[i + 1])
        # Semakin besar selisihnya, semakin jelas bahwa setelah K_i perbaikan mulai melambat.
        score = max(prev_drop - next_drop, 0.0)
        scores[int(ks_arr[i])] = score

    if not scores or max(scores.values()) <= 1e-12:
        return _jarak_garis_elbow(ks, inertias)

    # Jika ada skor sama, pilih K yang lebih kecil agar interpretasi tidak terlalu terpecah
    # pada data dengan jumlah objek sedikit.
    best_k = min(
        [k for k, v in scores.items() if abs(v - max(scores.values())) <= 1e-12]
    )
    return int(best_k), scores


def cari_siku_elbow(ks, inertias):
    """Menentukan titik siku dengan metode jarak maksimum ke garis awal-akhir.

    Nilai K dan WCSS diskalakan ke rentang yang sebanding terlebih dahulu supaya skala sumbu tidak
    memengaruhi hasil. Titik pertama dan terakhir tidak boleh dipilih sebagai
    siku karena keduanya hanya menjadi batas kurva.
    """
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)

    if len(x) < 3:
        return int(x[0])

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]], dtype=float)
    p2 = np.array([x_norm[-1], y_norm[-1]], dtype=float)
    garis = p2 - p1
    penyebut = np.linalg.norm(garis)

    if penyebut <= 1e-12:
        return int(x[1])

    jarak = []
    for xi, yi in zip(x_norm, y_norm):
        titik = np.array([xi, yi], dtype=float)
        d = abs(garis[0] * (p1[1] - titik[1]) - garis[1] * (p1[0] - titik[0])) / penyebut
        jarak.append(float(d))

    # Endpoint bukan kandidat siku.
    jarak[0] = -np.inf
    jarak[-1] = -np.inf
    nilai_maks = np.nanmax(jarak)
    indeks_calon = [i for i, d in enumerate(jarak) if np.isclose(d, nilai_maks)]
    elbow_idx = min(indeks_calon, key=lambda i: x[i])
    return int(x[elbow_idx])

def buat_kandidat_dari_elbow(elbow_k: int, ks: list, jumlah_minimal: int = 3, total_objek: int | None = None) -> list:
    """Menetapkan kandidat utama K = 2, 3, dan 4 sesuai arahan pembimbing.

    Elbow tetap dihitung pada seluruh rentang K sebagai bukti visual letak titik
    siku. Namun, validasi utama selalu membandingkan K=2, K=3, dan K=4 dengan
    Silhouette Score serta distribusi anggota cluster. Apabila jumlah objek tidak
    memungkinkan salah satu nilai K, hanya K yang valid yang digunakan.
    """
    ks_valid = sorted(int(k) for k in ks if int(k) >= 2)
    kandidat_dospem = [k for k in (2, 3, 4) if k in ks_valid]
    return kandidat_dospem


def format_ukuran_cluster(ukuran_cluster: list) -> str:
    """Menampilkan distribusi dengan label cluster agar angka tidak ambigu."""
    return ", ".join(
        f"C{i}= {int(jumlah)} anggota"
        for i, jumlah in enumerate(ukuran_cluster or [], start=1)
    )


def rasio_kepadatan_cluster(ukuran_cluster: list) -> float:
    """
    Mengukur keseimbangan distribusi anggota cluster.

    Rumus: jumlah anggota cluster terkecil / jumlah anggota cluster terbesar.
    Nilai mendekati 1 berarti distribusi anggota lebih seimbang; nilai mendekati 0
    berarti terdapat cluster yang jauh lebih kecil dibanding cluster lain.
    """
    ukuran = [int(x) for x in (ukuran_cluster or []) if int(x) >= 0]
    if not ukuran or max(ukuran) <= 0:
        return 0.0
    return float(min(ukuran) / max(ukuran))



def ada_cluster_tunggal(ukuran_cluster: list) -> bool:
    """Mengembalikan True jika ada cluster berisi satu anggota."""
    try:
        return any(int(x) <= 1 for x in (ukuran_cluster or []))
    except Exception:
        return False

def kualitas_silhouette_ringkas(score):
    """Label ringkas untuk tabel evaluasi K kandidat."""
    if score is None or pd.isna(score):
        return "Tidak dapat dihitung"
    score = float(score)
    if score >= 0.71:
        return "Kuat"
    if score >= 0.51:
        return "Baik"
    if score >= 0.26:
        return "Lemah"
    if score >= 0.00:
        return "Rendah / beririsan"
    return "Negatif / perlu dicermati"


def pilih_k_final_dari_kandidat(
    kandidat_k: list,
    ks: list,
    silhouettes_per_k: list,
    ukuran_cluster_all: list,
    elbow_k: int,
    min_cluster_size: int = 2,
    tie_tolerance: float = 1e-9,
) -> tuple[int, int | None, float | None, str]:
    """Memilih K final dari K=2, K=3, dan K=4 secara transparan.

    Alur mengikuti arahan pembimbing:
    1) kualitas pemisahan dibandingkan melalui Silhouette Score;
    2) distribusi anggota diperiksa agar tidak ada cluster tunggal;
    3) dari kandidat yang distribusinya layak, dipilih Silhouette tertinggi;
    4) rasio kepadatan hanya menjadi pemecah seri jika Silhouette sama.

    Dengan aturan ini, distribusi bukan sekadar informasi tambahan: kandidat
    dengan cluster beranggotakan satu tidak langsung ditetapkan sebagai K final.
    """
    kandidat_valid = []
    for k in kandidat_k:
        try:
            idx = ks.index(int(k))
            sil = float(silhouettes_per_k[idx])
            sizes = [int(x) for x in ukuran_cluster_all[idx]]
        except Exception:
            continue

        if not np.isfinite(sil) or not sizes:
            continue

        kandidat_valid.append({
            "K": int(k),
            "Silhouette": sil,
            "Ukuran_Cluster": sizes,
            "Min_Anggota": int(min(sizes)),
            "Max_Anggota": int(max(sizes)),
            "Rasio_Kepadatan": rasio_kepadatan_cluster(sizes),
            "Distribusi_Layak": int(min(sizes)) >= int(min_cluster_size),
        })

    if not kandidat_valid:
        return (
            int(elbow_k),
            None,
            None,
            "Silhouette kandidat tidak dapat dihitung. Periksa jumlah objek dan hasil preprocessing.",
        )

    # Kandidat dengan Silhouette tertinggi sebelum pemeriksaan distribusi.
    terbaik_silhouette_murni = max(
        kandidat_valid,
        key=lambda item: (
            item["Silhouette"],
            item["Rasio_Kepadatan"],
            item["Min_Anggota"],
            -item["K"],
        ),
    )

    kandidat_layak = [item for item in kandidat_valid if item["Distribusi_Layak"]]
    if kandidat_layak:
        skor_tertinggi_layak = max(item["Silhouette"] for item in kandidat_layak)
        kandidat_skor_seri = [
            item for item in kandidat_layak
            if abs(float(item["Silhouette"]) - float(skor_tertinggi_layak)) <= float(tie_tolerance)
        ]
        selected = max(
            kandidat_skor_seri,
            key=lambda item: (
                item["Rasio_Kepadatan"],
                item["Min_Anggota"],
                -item["K"],
            ),
        )

        if int(selected["K"]) == int(terbaik_silhouette_murni["K"]):
            note = (
                f"K={selected['K']} dipilih karena memiliki Silhouette Score tertinggi "
                f"di antara kandidat yang distribusinya layak (setiap cluster minimal "
                f"{min_cluster_size} anggota)."
            )
        else:
            note = (
                f"K={terbaik_silhouette_murni['K']} memiliki Silhouette tertinggi secara murni, "
                f"tetapi tidak dipilih karena membentuk cluster beranggotakan kurang dari "
                f"{min_cluster_size}. K={selected['K']} menjadi K final karena memiliki "
                f"Silhouette tertinggi di antara kandidat dengan distribusi anggota yang layak."
            )

        if len(kandidat_skor_seri) > 1:
            nilai_seri = ", ".join(f"K={item['K']}" for item in kandidat_skor_seri)
            note += (
                f" Silhouette seri secara numerik pada {nilai_seri}; rasio kepadatan "
                "digunakan sebagai pemecah seri."
            )
    else:
        selected = terbaik_silhouette_murni
        note = (
            f"Semua kandidat K=2, K=3, dan K=4 mempunyai cluster beranggotakan kurang dari "
            f"{min_cluster_size}. K={selected['K']} dipakai sementara karena Silhouette-nya "
            "tertinggi, tetapi hasil wajib diberi catatan keterbatasan distribusi anggota."
        )

    return (
        int(selected["K"]),
        int(terbaik_silhouette_murni["K"]),
        float(terbaik_silhouette_murni["Silhouette"]),
        note,
    )

def buat_tabel_validasi_k(
    ks: list,
    inertias: list,
    silhouettes_per_k: list,
    ukuran_cluster_all: list,
    kandidat_k: list,
    elbow_k: int,
    best_k: int,
    best_silhouette_k: int | None,
    min_cluster_size: int = 2,
) -> pd.DataFrame:
    """Menyusun tabel WCSS, Silhouette, dan kelayakan distribusi anggota."""
    rows = []
    kandidat_set = {int(x) for x in kandidat_k}

    for idx, k in enumerate(ks):
        sizes = [int(x) for x in ukuran_cluster_all[idx]] if idx < len(ukuran_cluster_all) else []
        sil = silhouettes_per_k[idx] if idx < len(silhouettes_per_k) else np.nan
        min_member = int(min(sizes)) if sizes else 0
        max_member = int(max(sizes)) if sizes else 0
        rasio = rasio_kepadatan_cluster(sizes)
        distribusi_layak = bool(sizes) and min_member >= int(min_cluster_size)

        if int(k) == int(best_k):
            status = "K final dipakai"
        elif int(k) in kandidat_set:
            status = "Kandidat validasi"
        else:
            status = "Referensi kurva WCSS"

        if int(k) == int(best_k):
            catatan = "Dipilih setelah menilai Silhouette Score dan kelayakan distribusi anggota."
        elif best_silhouette_k is not None and int(k) == int(best_silhouette_k) and not distribusi_layak:
            catatan = "Silhouette tertinggi secara murni, tetapi distribusi tidak layak karena ada cluster tunggal."
        elif int(k) in kandidat_set and not distribusi_layak:
            catatan = "Tidak layak sebagai pilihan utama karena ada cluster dengan anggota kurang dari 2."
        elif int(k) in kandidat_set:
            catatan = "Kandidat layak; dibandingkan berdasarkan Silhouette Score."
        else:
            catatan = "Hanya ditampilkan untuk melihat pola penurunan WCSS."

        rows.append({
            "K": int(k),
            "WCSS": round(float(inertias[idx]), 6),
            "Silhouette_Score": np.nan if pd.isna(sil) else round(float(sil), 6),
            "Kualitas_Silhouette": kualitas_silhouette_ringkas(sil),
            "Ukuran_Cluster": format_ukuran_cluster(sizes),
            "Min_Anggota": min_member,
            "Max_Anggota": max_member,
            "Rasio_Kepadatan": round(float(rasio), 3),
            "Distribusi_Layak": "Ya" if distribusi_layak else "Tidak",
            "Kandidat_K_2_3_4": "Ya" if int(k) in kandidat_set else "Tidak",
            "Silhouette_Tertinggi_Murni": (
                "Ya" if best_silhouette_k is not None and int(k) == int(best_silhouette_k) else "Tidak"
            ),
            "K_Dipakai": "Ya" if int(k) == int(best_k) else "Tidak",
            "Status": status,
            "Catatan_Pemilihan": catatan,
        })

    return pd.DataFrame(rows)

def buat_label_kategori(jumlah_cluster: int) -> list:
    if jumlah_cluster == 2:
        return ["Rendah", "Tinggi"]
    if jumlah_cluster == 3:
        return ["Rendah", "Sedang", "Tinggi"]
    if jumlah_cluster == 4:
        return ["Rendah", "Sedang Rendah", "Sedang Tinggi", "Tinggi"]
    if jumlah_cluster == 5:
        return ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]

    labels = []
    for i in range(jumlah_cluster):
        if i == 0:
            labels.append("Paling Rendah")
        elif i == jumlah_cluster - 1:
            labels.append("Paling Tinggi")
        else:
            labels.append(f"Kelompok Tengah {i}")
    return labels


def plot_silhouette(ax, X_scaled, labels, n_clusters=None):
    """Silhouette plot blok dengan informasi kepadatan anggota cluster.

    Fungsi ini sengaja membaca label cluster dari nilai unik pada `labels`, bukan
    memakai asumsi label selalu 0..K-1. Dengan begitu plot final tetap konsisten
    dengan nomor cluster hasil interpretasi (Cluster 1, Cluster 2, dst.) setelah
    label K-Means diurutkan ulang berdasarkan centroid harga.
    """
    labels = np.asarray(labels)
    unique_labels = sorted(np.unique(labels).tolist())
    n_clusters_aktual = len(unique_labels)

    if n_clusters_aktual < 2 or n_clusters_aktual >= len(X_scaled):
        ax.text(0.5, 0.5, "Silhouette tidak dapat dihitung", ha="center", va="center")
        ax.set_axis_off()
        return np.nan

    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    silhouette_avg = float(np.mean(sample_silhouette_values))

    y_lower = 10
    y_ticks = []
    y_labels = []

    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([0, len(X_scaled) + (n_clusters_aktual + 1) * 10])

    label_min = min(unique_labels)
    for idx_cluster, label_value in enumerate(unique_labels):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == label_value]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = int(ith_cluster_silhouette_values.shape[0])
        if size_cluster_i == 0:
            continue

        y_upper = y_lower + size_cluster_i
        y_range = np.arange(y_lower, y_upper)
        color = plt.cm.nipy_spectral(float(idx_cluster) / max(1, n_clusters_aktual))

        cluster_mean = float(np.mean(ith_cluster_silhouette_values))
        cluster_min = float(np.min(ith_cluster_silhouette_values))
        cluster_max = float(np.max(ith_cluster_silhouette_values))
        zero_width_cluster = np.allclose(ith_cluster_silhouette_values, 0.0, atol=1e-12)

        if zero_width_cluster:
            ax.fill_betweenx(
                y_range,
                -0.006,
                0.006,
                facecolor=color,
                edgecolor=color,
                alpha=0.45,
            )
            ax.text(
                0.025,
                y_lower + 0.5 * size_cluster_i,
                "silhouette = 0\n(cluster tunggal/tepi)",
                va="center",
                ha="left",
                fontsize=7.2,
            )
        else:
            ax.fill_betweenx(
                y_range,
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
            )

        ax.vlines(cluster_mean, y_lower, y_upper, colors="gray", linestyles=":", linewidth=1.2)

        # Jika label asli 0..K-1, tampilkan sebagai Cluster 1..K. Jika label final
        # sudah 1..K, tampilkan apa adanya agar sama dengan tabel hasil cluster.
        display_cluster = int(label_value) + 1 if label_min == 0 else int(label_value)
        y_ticks.append(y_lower + 0.5 * size_cluster_i)
        extra_label = " | singleton" if size_cluster_i == 1 else ""
        y_labels.append(f"C{display_cluster} | n={size_cluster_i} | mean={cluster_mean:.3f}{extra_label}")
        ax.text(
            -0.045,
            y_lower + 0.5 * size_cluster_i,
            f"min={cluster_min:.3f}\nmax={cluster_max:.3f}",
            va="center",
            ha="right",
            fontsize=7.5,
        )
        y_lower = y_upper + 10

    title_k = n_clusters if n_clusters is not None else n_clusters_aktual
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=1.6, label=f"Rata-rata total = {silhouette_avg:.4f}")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.55)
    ax.set_title(f"Silhouette Plot Kandidat K = {title_k} (avg = {silhouette_avg:.4f})", fontsize=10)
    ax.set_xlabel("Nilai Silhouette", fontsize=9)
    ax.set_ylabel("Cluster dan kepadatan anggota", fontsize=9)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(axis="x", labelsize=8)

    return silhouette_avg

# =========================================================
# FUNGSI KATEGORI SILHOUETTE (sesuai tabel interpretasi)
# =========================================================
def kategori_silhouette(x):
    if pd.isna(x):
        return "Tidak Dapat Dihitung"
    if x >= 0.71:
        return "Kuat"
    elif x >= 0.51:
        return "Baik"
    elif x >= 0.26:
        return "Lemah"
    else:
        return "Tidak Terstruktur"

def label_kategori_silhouette(x):
    if pd.isna(x):
        return "tidak dapat dihitung"
    if x >= 0.71:
        return "kuat"
    elif x >= 0.51:
        return "baik"
    elif x >= 0.26:
        return "lemah"
    else:
        return "tidak terstruktur"

def hitung_silhouette_aman(X, labels):
    labels = np.asarray(labels)
    n_label_unik = len(np.unique(labels))
    if n_label_unik < 2 or n_label_unik >= len(X):
        return np.nan
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return np.nan


def jalankan_kmeans_sklearn(X, k, random_state=RANDOM_STATE):
    """
    Menjalankan K-Means menggunakan class KMeans dari scikit-learn.

    Parameter dibuat eksplisit agar metode yang dipakai jelas saat dijelaskan pada
    laporan/sidang: init="k-means++", n_init=10, max_iter=300, dan random_state tetap.
    Fungsi ini mengembalikan label cluster, centroid hasil standardisasi, nilai WCSS/inertia,
    dan model KMeans yang sudah dilatih.
    """
    model = KMeans(
        n_clusters=int(k),
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    inertia = float(model.inertia_)
    return labels, centroids, inertia, model


def urutkan_label_cluster_berdasarkan_harga(labels, centroids, indeks_harga_rata2: int = 0):
    """Mengurutkan ulang label K-Means agar C1 selalu kelompok harga terendah.

    Label bawaan K-Means (0, 1, 2, dst.) bersifat arbitrer dan dapat berubah urutan.
    Agar angka C1, C2, C3 pada grafik, tabel, dan interpretasi tidak tertukar,
    centroid diurutkan berdasarkan fitur Harga_Rata2. Karena StandardScaler bersifat
    monoton, urutan pada skala standardisasi sama dengan urutan pada skala rupiah.

    Hasil label menggunakan nomor 1..K:
    C1 = centroid Harga_Rata2 paling rendah, dan CK = paling tinggi.
    """
    labels = np.asarray(labels, dtype=int)
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[0] == 0:
        return labels.copy(), centroids.copy(), {}

    urutan_lama = np.argsort(centroids[:, int(indeks_harga_rata2)])
    mapping = {int(label_lama): int(posisi_baru + 1) for posisi_baru, label_lama in enumerate(urutan_lama)}
    labels_baru = np.array([mapping[int(label)] for label in labels], dtype=int)
    centroids_baru = centroids[urutan_lama]
    return labels_baru, centroids_baru, mapping


def evaluasi_kandidat_lengkap(X_scaled: np.ndarray, nama_objek, k: int):
    """Menghasilkan audit lengkap satu kandidat K.

    Output mencakup:
    - label cluster yang diurutkan berdasarkan centroid Harga_Rata2;
    - nilai Silhouette tiap kabupaten/kota;
    - jumlah dan nama anggota setiap cluster;
    - sumber angka n, mean, min, dan max pada grafik;
    - rumus mean per cluster dan rata-rata total tertimbang.

    Fungsi ini dibuat supaya setiap angka pada grafik dapat ditelusuri sampai ke
    nama objek dan nilai Silhouette yang membentuknya.
    """
    labels_raw, centroids_scaled, inertia, model = jalankan_kmeans_sklearn(X_scaled, int(k))
    labels_urut, centroids_urut, mapping = urutkan_label_cluster_berdasarkan_harga(
        labels_raw, centroids_scaled, indeks_harga_rata2=0
    )

    sil_avg = hitung_silhouette_aman(X_scaled, labels_urut)
    if np.isfinite(sil_avg):
        sil_samples = silhouette_samples(X_scaled, labels_urut)
    else:
        sil_samples = np.full(len(X_scaled), np.nan, dtype=float)

    nama_objek = pd.Series(nama_objek).reset_index(drop=True).astype(str)
    kategori_labels = buat_label_kategori(int(k))
    kategori_map_local = {
        cluster_id: kategori_labels[cluster_id - 1]
        for cluster_id in range(1, int(k) + 1)
    }

    # Detail memakai nilai 6 desimal agar mean/min/max pada tabel dapat ditelusuri
    # dengan lebih presisi daripada label grafik yang hanya dibulatkan 3 desimal.
    detail_raw = pd.DataFrame({
        "Kabupaten/Kota": nama_objek,
        "Cluster": labels_urut,
        "Kategori": [kategori_map_local.get(int(c), "") for c in labels_urut],
        "Silhouette_Score": sil_samples,
    })
    detail = detail_raw.copy()
    detail["Silhouette_Score"] = pd.to_numeric(
        detail["Silhouette_Score"], errors="coerce"
    ).round(6)
    detail = detail.sort_values(
        ["Cluster", "Silhouette_Score", "Kabupaten/Kota"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    ringkasan_rows = []
    bukti_rows = []
    komponen_total = []

    for cluster_id in range(1, int(k) + 1):
        sub_raw = detail_raw[detail_raw["Cluster"] == cluster_id].copy()
        sub_raw = sub_raw.sort_values(
            ["Silhouette_Score", "Kabupaten/Kota"],
            ascending=[False, True],
        ).reset_index(drop=True)
        nilai_sil = pd.to_numeric(sub_raw["Silhouette_Score"], errors="coerce")
        n_anggota = int(len(sub_raw))
        nama_anggota = sub_raw["Kabupaten/Kota"].tolist()

        if nilai_sil.notna().any():
            mean_cluster = float(nilai_sil.mean())
            idx_min = int(nilai_sil.idxmin())
            idx_max = int(nilai_sil.idxmax())
            min_cluster = float(nilai_sil.loc[idx_min])
            max_cluster = float(nilai_sil.loc[idx_max])
            nama_min = str(sub_raw.loc[idx_min, "Kabupaten/Kota"])
            nama_max = str(sub_raw.loc[idx_max, "Kabupaten/Kota"])
            nilai_terms = [f"{float(v):.6f}" for v in nilai_sil.dropna().tolist()]
            rumus_mean = f"({' + '.join(nilai_terms)}) / {n_anggota} = {mean_cluster:.6f}"
            daftar_nilai = "; ".join(
                f"{row['Kabupaten/Kota']}={float(row['Silhouette_Score']):.6f}"
                for _, row in sub_raw.iterrows()
                if pd.notna(row["Silhouette_Score"])
            )
            komponen_total.append((n_anggota, mean_cluster, cluster_id))
        else:
            mean_cluster = np.nan
            min_cluster = np.nan
            max_cluster = np.nan
            nama_min = "Tidak tersedia"
            nama_max = "Tidak tersedia"
            rumus_mean = "Tidak dapat dihitung"
            daftar_nilai = "Tidak tersedia"

        ringkasan_rows.append({
            "Cluster": f"C{cluster_id}",
            "Kategori": kategori_map_local.get(cluster_id, ""),
            "Jumlah_Anggota_n": n_anggota,
            "Nama_Anggota": ", ".join(nama_anggota),
            "Mean_Silhouette": round(mean_cluster, 6) if np.isfinite(mean_cluster) else np.nan,
            "Min_Silhouette": round(min_cluster, 6) if np.isfinite(min_cluster) else np.nan,
            "Anggota_dengan_Min": nama_min,
            "Max_Silhouette": round(max_cluster, 6) if np.isfinite(max_cluster) else np.nan,
            "Anggota_dengan_Max": nama_max,
        })

        bukti_rows.append({
            "Cluster": f"C{cluster_id}",
            "Kategori": kategori_map_local.get(cluster_id, ""),
            "Asal_n": f"{n_anggota} nama: {', '.join(nama_anggota)}",
            "Nilai_per_Anggota": daftar_nilai,
            "Rumus_Mean": rumus_mean,
            "Asal_Min": (
                f"{nama_min} = {min_cluster:.6f}"
                if np.isfinite(min_cluster) else "Tidak tersedia"
            ),
            "Asal_Max": (
                f"{nama_max} = {max_cluster:.6f}"
                if np.isfinite(max_cluster) else "Tidak tersedia"
            ),
        })

    ringkasan = pd.DataFrame(ringkasan_rows)
    bukti = pd.DataFrame(bukti_rows)
    counts = ringkasan["Jumlah_Anggota_n"].astype(int).tolist()

    total_n = int(sum(n for n, _, _ in komponen_total))
    if total_n > 0:
        pembilang_total = float(sum(n * mean for n, mean, _ in komponen_total))
        rata_total_bobot = pembilang_total / total_n
        bagian_rumus = " + ".join(
            f"({n} × {mean:.6f})" for n, mean, _ in komponen_total
        )
        rumus_total = (
            f"Rata-rata total = [{bagian_rumus}] / {total_n} "
            f"= {rata_total_bobot:.6f}"
        )
    else:
        rata_total_bobot = np.nan
        rumus_total = "Rata-rata total tidak dapat dihitung."

    return {
        "labels": labels_urut,
        "centroids": centroids_urut,
        "inertia": float(inertia),
        "model": model,
        "mapping": mapping,
        "silhouette_avg": float(sil_avg) if np.isfinite(sil_avg) else np.nan,
        "silhouette_samples": sil_samples,
        "counts": counts,
        "ringkasan": ringkasan,
        "detail": detail,
        "bukti": bukti,
        "rumus_total": rumus_total,
        "silhouette_avg_tertimbang": (
            float(rata_total_bobot) if np.isfinite(rata_total_bobot) else np.nan
        ),
    }


@st.cache_data(show_spinner="Mencari K kandidat (Elbow) dan evaluasi Silhouette...")
def cached_elbow_silhouette_search(X_scaled: np.ndarray, ks_tuple: tuple):
    """
    Menjalankan K-Means scikit-learn untuk setiap K pada ks_tuple.

    Elbow/WCSS digunakan untuk membaca titik siku. Silhouette Score dihitung untuk
    setiap K, sedangkan jumlah anggota disimpan menurut label yang telah diurutkan
    berdasarkan centroid Harga_Rata2: C1 terendah sampai CK tertinggi.
    """
    inertias, silhouettes_per_k = [], []
    ukuran_cluster_min, ukuran_cluster_all = [], []
    for k in ks_tuple:
        labels_raw, centroids_k, inertia_k, _ = jalankan_kmeans_sklearn(X_scaled, k)
        labels_k, _, _ = urutkan_label_cluster_berdasarkan_harga(labels_raw, centroids_k, indeks_harga_rata2=0)
        inertias.append(inertia_k)
        # Nilai Silhouette tidak berubah akibat penggantian nomor label.
        silhouettes_per_k.append(hitung_silhouette_aman(X_scaled, labels_k))
        sizes_k = [int(np.sum(labels_k == cluster_id)) for cluster_id in range(1, int(k) + 1)]
        ukuran_cluster_all.append(sizes_k)
        ukuran_cluster_min.append(int(min(sizes_k)))
    return inertias, silhouettes_per_k, ukuran_cluster_all, ukuran_cluster_min

# =========================================================
# UI HEADER
# =========================================================
st.title("🍬 Clustering Harga Gula Pasir (K-Means)")
st.caption(
    "Data mentah → cleaning → transformasi fitur → eksplorasi → standardisasi → "
    "Elbow Method → pemilihan K → clustering → evaluasi → visualisasi → interpretasi"
)

with st.sidebar:
    st.header("📁 Input Data")
    file = st.file_uploader("Upload Data.xlsx", type=["xlsx"])

    st.markdown("---")
    st.header("🧹 Data Cleaning")
    outlier_mode = "Verifikasi manual pada tabel outlier"
    st.write("Outlier: **dideteksi, diperiksa asalnya, lalu diberi status verifikasi**")
    st.caption(
        "Nilai yang valid dipertahankan. Nilai yang terbukti salah catat dapat ditandai untuk dihapus dari analisis, "
        "disertai alasan verifikasi."
    )

    st.markdown("---")
    st.header("📊 K-Means")
    st.text("Scaler: StandardScaler")
    st.caption("Kurva WCSS ditampilkan sebagai Elbow; keputusan membandingkan K=2, K=3, dan K=4 melalui Silhouette serta distribusi anggota.")
    st.caption(f"Versi aplikasi: {APP_VERSION}")

    if st.button("🔄 Bersihkan cache & muat ulang"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


period_mode = "Harian"
fill_missing = True
use_seed = 42

if not file:
    st.info("Upload **Data.xlsx** dulu. Jalankan dengan: `streamlit run app.py`")
    st.stop()

# =========================================================
# LOAD EXCEL + PICK SHEET / AUTO-BUILD DATA MENTAH MITRA
# =========================================================
try:
    file_bytes = file.getvalue()
    xl = get_excel_file(file_bytes)
    sheet_names = xl.sheet_names
except Exception as e:
    st.error("Gagal membaca file Excel. Pastikan file .xlsx valid.")
    st.exception(e)
    st.stop()

sheet_harian_mitra = [s for s in sheet_names if is_sheet_harian_mitra(s)]
format_mitra_harian = ("Data mentah" not in sheet_names) and (len(sheet_harian_mitra) > 0)

if format_mitra_harian:
    with st.expander("1) Data Mentah (RAW)", expanded=False):
        st.write(
        f"Format mitra terdeteksi: **{len(sheet_harian_mitra)} sheet harian** diinput ke dalam sistem."
    )

        raw = cached_parse_all_daily_sheets(file_bytes, tuple(sheet_harian_mitra))

        st.dataframe(raw.head(15), use_container_width=True, hide_index=True)

        st.download_button(
        "Download Data Mentah (RAW) - CSV",
        data=df_to_csv_download(raw),
        file_name="data_mentah_raw.csv",
        mime="text/csv"
    )

else:
    default_sheet = "Data mentah" if "Data mentah" in sheet_names else sheet_names[0]
    sheet_choice = st.selectbox("Pilih sheet:", sheet_names, index=sheet_names.index(default_sheet))

    try:
        raw = cached_read_excel_sheet(file_bytes, sheet_choice)
    except Exception as e:
        st.error(f"Gagal membaca sheet: {sheet_choice}")
        st.exception(e)
        st.stop()

    with st.expander("1) Data Mentah (RAW)", expanded=False):
        st.write(f"Sheet dipakai: **{sheet_choice}**")
        st.dataframe(raw.head(15), use_container_width=True, hide_index=True)

        st.download_button(
        "Download Data Mentah (RAW) - CSV",
        data=df_to_csv_download(raw),
        file_name="data_mentah_raw.csv",
        mime="text/csv"
    )

# =========================================================
# STRUKTURISASI DATA (proses internal, dibutuhkan sebelum data
# cleaning bisa berjalan -- TIDAK ditampilkan sebagai bagian dari
# tahap "1) Data Mentah (RAW)" karena tahap itu murni untuk input
# data mentah ke dalam sistem, bukan untuk mengolah data)
# =========================================================
if format_mitra_harian:
    df_wide = cached_build_data_mentah_mitra(file_bytes, tuple(sheet_names))
    if df_wide.empty:
        st.error("Data mentah tidak berhasil dibentuk dari file mitra. Cek struktur sheet harian.")
        st.stop()
else:
    header_row = find_header_row(raw)
    df_wide = build_table_from_raw(raw, header_row)

if "Tanggal" not in df_wide.columns:
    st.error("Kolom **Tanggal** tidak ditemukan. Pastikan header ada Tanggal/tangga/tgl.")
    st.stop()

# =========================================================
# 2) DATA CLEANING
# =========================================================
st.markdown("---")
with st.expander("2) Data Cleaning", expanded=False):

    st.markdown("""
### Jenis Data Cleaning / Penyaringan yang Digunakan

Ada **empat jenis pembersihan/penyaringan data** yang diterapkan pada tahap ini:

| No | Jenis Cleaning | Keterangan |
|---|---|---|
| 1 | **Filter komoditi Gula Pasir** | Data mentah yang diinput bisa berisi banyak komoditi sekaligus (beras, minyak goreng, daging, telur, gula pasir, dsb). Baris yang **bukan** Gula Pasir disaring/dibuang, karena penelitian ini hanya berfokus pada Gula Pasir. |
| 2 | **Penghapusan data tidak valid** | Baris dengan **Tanggal** atau **Harga** yang kosong/tidak terbaca (`NaN`) dibuang, karena tidak bisa dipakai untuk perhitungan statistik maupun clustering. |
| 3 | **Penghapusan data duplikat** | Baris dengan kombinasi **Tanggal + Kabupaten/Kota** yang sama (tercatat lebih dari satu kali) dihapus, hanya baris terakhir yang dipertahankan (`keep="last"`). |
| 4 | **Deteksi outlier (metode IQR)** | Nilai harga yang berada di luar rentang **Q1 - 1.5×IQR** sampai **Q3 + 1.5×IQR** dideteksi dan dihitung jumlahnya. |
""")

    st.markdown("##### Proses Cleaning (dijalankan berurutan dari data mentah)")

    # =========================================================
    # WIDE -> LONG (seluruh komoditi, belum difilter)
    # =========================================================
    # Melt dilakukan LEBIH DAHULU, sebelum komoditi difilter, agar baseline
    # "sebelum cleaning" benar-benar mewakili seluruh data mentah yang diinput
    # (bukan data yang sudah diam-diam difilter duluan). Dengan begitu, tabel
    # perbandingan di bawah akan menunjukkan perbedaan nyata pada setiap langkah,
    # bukan sekadar membandingkan data yang sudah sama sejak awal.
    drop_cols = [c for c in ["No", "HET_HA"] if c in df_wide.columns]
    df_wide_clean = df_wide.drop(columns=drop_cols, errors="ignore").copy()

    id_vars = [c for c in ["Tanggal", "Jenis_Komoditi"] if c in df_wide_clean.columns]
    value_cols = [c for c in df_wide_clean.columns if c not in id_vars]
    df_long_all = df_wide_clean.melt(
    id_vars=id_vars,
    value_vars=value_cols,
    var_name="Kabupaten",
    value_name="Harga"
)

    df_long_all["Tanggal"] = pd.to_datetime(df_long_all["Tanggal"], errors="coerce", dayfirst=True)
    df_long_all["Kabupaten"] = df_long_all["Kabupaten"].astype(str).str.strip().str.title()
    df_long_all["Harga"] = to_numeric_safe(df_long_all["Harga"])

    # ---- SNAPSHOT DATA MENTAH (SELURUH KOMODITI, SEBELUM APAPUN DIBERSIHKAN) ----
    df_sebelum_cleaning = df_long_all.copy()
    jumlah_awal = len(df_long_all)

    # =========================================================
    # LANGKAH 1: FILTER KOMODITI GULA PASIR
    # =========================================================
    if "Jenis_Komoditi" in df_long_all.columns:
        mask_gula_pasir = (
        df_long_all["Jenis_Komoditi"].astype(str).str.lower().str.contains("gula pasir|gula kristal putih|gkp|gula", na=False)
    )
        daftar_komoditi_terdeteksi = sorted(df_long_all["Jenis_Komoditi"].dropna().astype(str).unique().tolist())

        if mask_gula_pasir.any():
            # Jenis_Komoditi SENGAJA DIPERTAHANKAN (bukan dibuang) supaya
            # tabel hasil cleaning tetap punya 4 atribut: Tanggal, Jenis_Komoditi,
            # Kabupaten, Harga -- bukan cuma 3 (Tanggal, Kabupaten, Harga).
            df_long = df_long_all.loc[mask_gula_pasir].reset_index(drop=True)
            # Standarisasi label supaya konsisten (data mentah bisa berisi variasi
            # penulisan seperti "Gula Kristal Putih", "GKP", dll -- semuanya
            # sudah tersaring oleh mask_gula_pasir di atas, jadi aman diseragamkan).
            df_long["Jenis_Komoditi"] = "Gula Pasir"
        else:
            st.error(
            f"Tidak ditemukan baris dengan komoditi 'Gula Pasir' pada data ini. "
            f"Komoditi yang terdeteksi: {', '.join(daftar_komoditi_terdeteksi)}. "
            f"Periksa kembali nama komoditi pada kolom Jenis_Komoditi."
        )
            st.stop()
    else:
        daftar_komoditi_terdeteksi = ["Gula Pasir"]
        df_long = df_long_all.copy()

    jumlah_setelah_filter_komoditi = len(df_long)
    dibuang_bukan_gula = jumlah_awal - jumlah_setelah_filter_komoditi

    st.markdown("**Langkah 1 — Filter Komoditi Gula Pasir**")
    if dibuang_bukan_gula > 0:
        st.info(
        f"🍬 Terdeteksi **{len(daftar_komoditi_terdeteksi)} jenis komoditi** pada data mentah: "
        f"{', '.join(daftar_komoditi_terdeteksi)}. Sebanyak **{dibuang_bukan_gula:,} baris komoditi lain** "
        f"disaring/dibuang, menyisakan **{jumlah_setelah_filter_komoditi:,} baris Gula Pasir** "
        f"(dari total {jumlah_awal:,} baris data mentah)."
    )
    else:
        st.info(
        f"Data mentah yang diinput sudah berisi Gula Pasir saja "
        f"({jumlah_setelah_filter_komoditi:,} baris), tidak ada komoditi lain yang perlu disaring."
    )

    # =========================================================
    # LANGKAH 2: PENGHAPUSAN DATA TIDAK VALID
    # =========================================================
    tanggal_tidak_terbaca = int(df_long["Tanggal"].isna().sum())
    harga_kosong = int(df_long["Harga"].isna().sum())

    df_long = df_long.dropna(subset=["Tanggal", "Harga"]).copy()
    jumlah_setelah_drop_na = len(df_long)
    dropped_invalid = jumlah_setelah_filter_komoditi - jumlah_setelah_drop_na

    st.markdown("**Langkah 2 — Penghapusan Data Tidak Valid**")
    st.write(
    f"Tanggal tidak terbaca: **{tanggal_tidak_terbaca:,} baris**, Harga kosong/NaN: **{harga_kosong:,} baris**. "
    f"Total baris tidak valid yang dibuang: **{dropped_invalid:,} baris** "
    f"(dari {jumlah_setelah_filter_komoditi:,} menjadi {jumlah_setelah_drop_na:,} baris)."
)

    # =========================================================
    # LANGKAH 3: PENGHAPUSAN DATA DUPLIKAT
    # =========================================================
    duplikat_awal = int(df_long.duplicated(subset=["Tanggal", "Kabupaten"], keep="last").sum())

    df_long = df_long.drop_duplicates(subset=["Tanggal", "Kabupaten"], keep="last").copy()
    dropped_duplikat = jumlah_setelah_drop_na - len(df_long)
    jumlah_setelah_duplikat = len(df_long)

    st.markdown("**Langkah 3 — Penghapusan Data Duplikat**")
    st.write(
    f"Terdeteksi **{duplikat_awal:,} baris** dengan kombinasi Tanggal + Kabupaten yang sama. "
    f"Baris duplikat yang dibuang: **{dropped_duplikat:,} baris** "
    f"(dari {jumlah_setelah_drop_na:,} menjadi {len(df_long):,} baris)."
)

    # =========================================================
    # LANGKAH 4: DETEKSI, VERIFIKASI, DAN PENANGANAN OUTLIER (IQR)
    # =========================================================
    # IQR hanya mendeteksi kandidat nilai ekstrem. Keputusan mempertahankan atau
    # menghapus dilakukan setelah pengguna memeriksa tanggal, kabupaten/kota,
    # nilai harga, dan alasan pada sumber data.
    _, outlier_terdeteksi, outlier_low, outlier_high = iqr_clip(df_long["Harga"])
    q1_outlier = df_long["Harga"].quantile(0.25)
    q3_outlier = df_long["Harga"].quantile(0.75)
    iqr_outlier = q3_outlier - q1_outlier

    if pd.isna(outlier_low) or pd.isna(outlier_high):
        mask_outlier = pd.Series(False, index=df_long.index)
    else:
        mask_outlier = (df_long["Harga"] < outlier_low) | (df_long["Harga"] > outlier_high)

    df_outlier_rows = df_long.loc[mask_outlier].copy()
    outlier_cnt = int(mask_outlier.sum())
    baris_dibuang_outlier = 0
    jumlah_belum_verifikasi = outlier_cnt
    jumlah_valid = 0
    jumlah_salah_catat = 0

    st.markdown("**Langkah 4 — Deteksi, Verifikasi, dan Penanganan Outlier (IQR)**")
    if pd.isna(outlier_low) or pd.isna(outlier_high):
        st.warning(
            "IQR bernilai 0 atau tidak dapat dihitung, sehingga batas outlier tidak terbentuk. "
            "Tidak ada baris yang diubah atau dihapus."
        )
        tindakan_outlier = "batas IQR tidak terbentuk; tidak ada tindakan"
    elif df_outlier_rows.empty:
        st.success(
            f"Batas IQR otomatis adalah **Rp {outlier_low:,.0f} sampai Rp {outlier_high:,.0f}**. "
            "Tidak ditemukan baris di luar batas tersebut."
        )
        tindakan_outlier = "tidak ada outlier"
    else:
        df_outlier_rows["Posisi"] = np.where(
            df_outlier_rows["Harga"] < outlier_low,
            "Di bawah batas bawah",
            "Di atas batas atas",
        )
        df_outlier_rows["ID_Baris"] = df_outlier_rows.index.astype(int)
        df_outlier_rows["Status_Verifikasi"] = "Belum diverifikasi"
        df_outlier_rows["Alasan_Verifikasi"] = ""
        df_outlier_rows["Tindakan"] = "Pertahankan sementara"

        st.info(
            f"Batas IQR otomatis: **Rp {outlier_low:,.0f} sampai Rp {outlier_high:,.0f}**. "
            f"Terdeteksi **{outlier_cnt:,} baris** kandidat outlier. IQR hanya menandai nilai ekstrem; "
            "status akhirnya harus ditentukan setelah memeriksa sumber data."
        )

        st.markdown("##### Tabel Verifikasi Outlier")
        st.caption(
            "Untuk setiap baris, pilih status **Valid/kondisi riil**, **Salah catat**, atau "
            "**Belum diverifikasi**. Isi alasan. Baris hanya dihapus bila statusnya Salah catat "
            "dan tindakannya Hapus dari analisis."
        )

        kolom_editor = [
            c for c in [
                "ID_Baris", "Tanggal", "Jenis_Komoditi", "Kabupaten", "Harga", "Posisi",
                "Status_Verifikasi", "Alasan_Verifikasi", "Tindakan"
            ] if c in df_outlier_rows.columns
        ]
        hasil_verifikasi = st.data_editor(
            df_outlier_rows[kolom_editor].sort_values(["Kabupaten", "Tanggal"]),
            use_container_width=True,
            hide_index=True,
            disabled=["ID_Baris", "Tanggal", "Jenis_Komoditi", "Kabupaten", "Harga", "Posisi"],
            column_config={
                "Status_Verifikasi": st.column_config.SelectboxColumn(
                    "Status Verifikasi",
                    options=["Belum diverifikasi", "Valid/kondisi riil", "Salah catat"],
                ),
                "Tindakan": st.column_config.SelectboxColumn(
                    "Tindakan",
                    options=["Pertahankan sementara", "Pertahankan", "Hapus dari analisis"],
                ),
                "Alasan_Verifikasi": st.column_config.TextColumn(
                    "Alasan Verifikasi",
                    help="Contoh: sesuai dokumen sumber; lonjakan harga riil; salah input angka.",
                ),
            },
            key="editor_verifikasi_outlier",
        )

        status = hasil_verifikasi["Status_Verifikasi"].astype(str)
        tindakan = hasil_verifikasi["Tindakan"].astype(str)
        alasan = hasil_verifikasi["Alasan_Verifikasi"].fillna("").astype(str).str.strip()

        jumlah_belum_verifikasi = int((status == "Belum diverifikasi").sum())
        jumlah_valid = int((status == "Valid/kondisi riil").sum())
        jumlah_salah_catat = int((status == "Salah catat").sum())

        mask_hapus = (status == "Salah catat") & (tindakan == "Hapus dari analisis")
        id_hapus = hasil_verifikasi.loc[mask_hapus, "ID_Baris"].astype(int).tolist()
        baris_dibuang_outlier = len(id_hapus)
        if id_hapus:
            df_long = df_long.drop(index=id_hapus, errors="ignore").copy()

        mask_status_tanpa_alasan = (status != "Belum diverifikasi") & (alasan == "")
        if mask_status_tanpa_alasan.any():
            st.warning(
                f"Ada **{int(mask_status_tanpa_alasan.sum())} baris** yang sudah diberi status tetapi belum "
                "memiliki alasan verifikasi. Isi alasan agar keputusan dapat dipertanggungjawabkan."
            )

        mask_salah_tapi_tidak_hapus = (status == "Salah catat") & (tindakan != "Hapus dari analisis")
        if mask_salah_tapi_tidak_hapus.any():
            st.warning(
                f"Ada **{int(mask_salah_tapi_tidak_hapus.sum())} baris** berstatus Salah catat tetapi belum "
                "dipilih Hapus dari analisis. Periksa kembali konsistensi status dan tindakan."
            )

        ringkasan_verifikasi = pd.DataFrame({
            "Status": ["Belum diverifikasi", "Valid/kondisi riil", "Salah catat", "Dihapus dari analisis"],
            "Jumlah_Baris": [
                jumlah_belum_verifikasi,
                jumlah_valid,
                jumlah_salah_catat,
                baris_dibuang_outlier,
            ],
        })
        st.dataframe(ringkasan_verifikasi, use_container_width=True, hide_index=True)

        if jumlah_belum_verifikasi > 0:
            st.warning(
                f"Masih ada **{jumlah_belum_verifikasi} baris outlier yang belum diverifikasi**. "
                "Hasil clustering dapat dijalankan, tetapi status ini harus diselesaikan sebelum hasil final dilaporkan."
            )
        else:
            st.success("Semua baris outlier sudah memiliki status verifikasi.")

        st.download_button(
            "⬇️ Unduh hasil verifikasi outlier (CSV)",
            data=df_to_csv_download(hasil_verifikasi),
            file_name="hasil_verifikasi_outlier.csv",
            mime="text/csv",
            key="download_hasil_verifikasi_outlier",
        )

        tindakan_outlier = (
            f"{outlier_cnt:,} terdeteksi; {jumlah_valid:,} valid; "
            f"{jumlah_salah_catat:,} salah catat; {baris_dibuang_outlier:,} dihapus; "
            f"{jumlah_belum_verifikasi:,} belum diverifikasi"
        )

    if st.checkbox("Dari mana batas IQR diperoleh?", key="show_iqr_formula"):
        with st.container():
            st.markdown(
                f"""
| Komponen | Rumus | Nilai |
|---|---|---:|
| Q1 | Kuartil 25% | Rp {q1_outlier:,.0f} |
| Q3 | Kuartil 75% | Rp {q3_outlier:,.0f} |
| IQR | Q3 − Q1 | Rp {iqr_outlier:,.0f} |
| Batas bawah | Q1 − 1,5 × IQR | Rp {outlier_low:,.0f} |
| Batas atas | Q3 + 1,5 × IQR | Rp {outlier_high:,.0f} |

IQR berfungsi sebagai **alat deteksi**, bukan keputusan otomatis. Nilai ekstrem yang sesuai sumber
atau merepresentasikan kondisi nyata dipertahankan. Nilai yang terbukti salah catat dapat dihapus
dari analisis setelah status dan alasan verifikasinya dicatat.
"""
            )

    # Snapshot data bersih dibuat setelah keputusan verifikasi outlier diterapkan.
    df_sesudah_cleaning = df_long.copy()

    st.success(
    f"Data cleaning selesai. Aplikasi berhasil membentuk **Data Bersih Gula Pasir** "
    f"sebanyak **{len(df_long):,} baris** dari **{df_long['Kabupaten'].nunique():,} kabupaten/kota**."
)

    total_dibuang = dibuang_bukan_gula + dropped_invalid + dropped_duplikat

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data mentah (semua komoditi)", f"{jumlah_awal:,}")
    c2.metric("Bukan Gula Pasir dibuang", f"{dibuang_bukan_gula:,}")
    c3.metric("Data tidak valid dibuang", f"{dropped_invalid:,}")
    c4.metric("Data duplikat dibuang", f"{dropped_duplikat:,}")
    c5.metric("Jumlah data akhir", f"{len(df_long):,}")

    # =========================================================
    # TABEL PERBANDINGAN (FUNNEL PER TAHAP)
    # =========================================================
    # Setiap baris tabel adalah SATU TAHAP dalam urutan proses cleaning yang
    # sebenarnya (bukan dibandingkan satu-satu secara terpisah), sehingga jumlah
    # baris pada tiap tahap benar-benar mencerminkan efek dari langkah sebelumnya
    # secara berantai/kumulatif.
    ringkasan_cleaning = pd.DataFrame({
    "Tahap": [
        "0. Data Mentah (RAW, semua komoditi)",
        "1. Setelah Filter Gula Pasir",
        "2. Setelah Hapus Data Tidak Valid",
        "3. Setelah Hapus Duplikat",
        f"4. Data Bersih Akhir ({tindakan_outlier})",
    ],
    "Jumlah Baris": [
        f"{jumlah_awal:,}",
        f"{jumlah_setelah_filter_komoditi:,}",
        f"{jumlah_setelah_drop_na:,}",
        f"{jumlah_setelah_duplikat:,}",
        f"{len(df_sesudah_cleaning):,}",
    ],
    "Baris Dibuang pada Tahap Ini": [
        "-",
        f"{dibuang_bukan_gula:,}",
        f"{dropped_invalid:,}",
        f"{dropped_duplikat:,}",
        f"{baris_dibuang_outlier:,} (tindakan: {outlier_mode})",
    ],
})

    st.markdown("### Tabel Perbandingan Sebelum vs Sesudah Cleaning (per Tahap)")
    st.caption(
    "Perbandingan dihitung dari data hasil input (upload) kamu sendiri secara berurutan/berantai — "
    "setiap tahap memakai hasil dari tahap sebelumnya, bukan angka tetap/hardcode."
)
    st.dataframe(ringkasan_cleaning, use_container_width=True, hide_index=True)


    # REMOVED comparison table per request
    st.markdown("### Kabupaten/Kota dalam Dataset")

    daftar_kabupaten = sorted(
    df_long["Kabupaten"]
    .dropna()
    .unique()
    .tolist()
)
    st.write(", ".join(daftar_kabupaten))

    st.markdown("### Data Cleaning")

    # Susun ulang urutan kolom jadi 4 atribut final:
    # Tanggal, Jenis_Komoditi, Kabupaten, Harga
    kolom_final = [c for c in ["Tanggal", "Jenis_Komoditi", "Kabupaten", "Harga"] if c in df_long.columns]
    df_long = df_long[kolom_final]

    st.dataframe(
    df_long.sort_values(["Kabupaten", "Tanggal"]),
    use_container_width=True,
    hide_index=True,
    height=500
)

    st.download_button(
    "Download Data Bersih Gula Pasir (CSV)",
    data=df_to_csv_download(df_long.sort_values(["Kabupaten", "Tanggal"])),
    file_name="data_bersih_gula_pasir.csv",
    mime="text/csv"
)

    # ---- Rentang tanggal: penjelasan, bukan step terpisah ----
    df_long_f = df_long.copy()
    kabupaten_filter = sorted(df_long_f["Kabupaten"].dropna().unique().tolist())
    min_date = df_long_f["Tanggal"].min()
    max_date = df_long_f["Tanggal"].max()
    start_date = min_date.date()
    end_date = max_date.date()

    st.markdown(f"""
>**Penjelasan:**

Setelah header berhasil dideteksi dan tabel dirapikan, data masih berupa **format lebar (wide)**
di mana setiap kabupaten/kota menjadi satu kolom tersendiri. Sebelum dipakai untuk clustering,
data diubah ke **format panjang (long)** lalu dilakukan **data cleaning** (lihat jenis cleaning
dan tabel perbandingan sebelum/sesudah di atas).

Perbandingan cleaning ditampilkan berdasarkan **jumlah baris dan jumlah masalah data**,
bukan berdasarkan harga minimum, maksimum, atau standar deviasi.

>**Rentang Tanggal dan Cakupan Data**

Data hasil cleaning mencakup rentang tanggal dari **{start_date}** sampai **{end_date}**,
dengan total **{len(df_long_f):,}** baris data pada **{len(kabupaten_filter)}** kabupaten/kota.
Seluruh rentang tanggal ini digunakan pada proses clustering (tidak difilter sebagian) agar hasil
pengelompokan konsisten dan tidak berubah-ubah karena pilihan rentang waktu tertentu.

**Batasan masalah:** data sumber bersifat harian, namun clustering tidak dilakukan untuk tiap tanggal
secara terpisah, melainkan berdasarkan **ringkasan statistik harga** setiap kabupaten/kota selama
seluruh periode pengamatan (lihat tahap Transformasi Fitur).
""")

    # =========================================================
    # 3) AGREGASI DATA (HARIAN -> PER KABUPATEN/KOTA)
    # =========================================================
st.markdown("---")
with st.expander("3) Agregasi Data (Data Harian → Per Kabupaten/Kota)", expanded=False):

    st.write(
    "Data hasil cleaning (tahap sebelumnya) masih berbentuk **data harian**: satu baris mewakili "
    "satu kombinasi Tanggal + Kabupaten/Kota. Objek yang ingin dikelompokkan pada penelitian ini "
    "adalah **kabupaten/kota**, bukan tanggal, sehingga data harian tersebut perlu **diagregasi** "
    "terlebih dahulu menjadi satu baris per kabupaten/kota sebelum dipakai untuk membentuk fitur K-Means."
)

    st.markdown(
    """
**Cara agregasi:** seluruh baris harian dikelompokkan berdasarkan kolom **Kabupaten**
(`groupby("Kabupaten")`), lalu untuk setiap kelompok dihitung beberapa ringkasan statistik dari
kolom **Harga** selama seluruh periode pengamatan:

| Ringkasan | Fungsi Agregasi |
|---|---|
| Harga_Rata2 | rata-rata (`mean`) seluruh harga harian kabupaten tersebut |
| Harga_Min | nilai harga terendah (`min`) yang pernah tercatat |
| Harga_Max | nilai harga tertinggi (`max`) yang pernah tercatat |
| Harga_Std | standar deviasi (`std`) harga; mengukur besar-kecilnya fluktuasi harga |
| Harga_Skewness | kemiringan (`skew`) distribusi harga; mengukur arah kecondongan sebaran harga |

Kelima ringkasan di atas adalah **fitur yang dipakai untuk K-Means**. Harga_Rata2, Harga_Min, dan
Harga_Max mengikuti pedoman bimbingan; Harga_Std dan Harga_Skewness ditambahkan agar fitur juga
menangkap **pola fluktuasi dan bentuk sebaran harga** tiap kabupaten/kota, bukan hanya titik
rata-rata/minimum/maksimum saja.

Hasilnya: dari data harian dengan ribuan baris, menjadi **satu tabel ringkas** dengan jumlah baris
sebanyak jumlah kabupaten/kota yang ada.
"""
)

    df_agregasi = (
    df_long_f.groupby("Kabupaten", as_index=False)
    .agg(
        Harga_Rata2=("Harga", "mean"),
        Harga_Min=("Harga", "min"),
        Harga_Max=("Harga", "max"),
        Harga_Std=("Harga", "std"),
        Harga_Skewness=("Harga", lambda x: x.skew()),
        Jumlah_Data_Harian=("Harga", "count"),
    )
)
    # Kabupaten dengan data harian sangat sedikit (< 3 baris) tidak punya std/skewness
    # yang valid secara statistik (NaN); diisi 0 agar tetap bisa dipakai K-Means, dengan
    # makna "variasi harga tidak terukur karena data historis terlalu sedikit".
    df_agregasi["Harga_Std"] = df_agregasi["Harga_Std"].fillna(0)
    df_agregasi["Harga_Skewness"] = df_agregasi["Harga_Skewness"].fillna(0)

    # Tanggal awal/akhir hanya untuk informasi cakupan periode, bukan fitur clustering.
    df_tanggal = (
        df_long_f.sort_values(["Kabupaten", "Tanggal"])
        .groupby("Kabupaten", as_index=False)
        .agg(
            Tanggal_Awal=("Tanggal", "first"),
            Tanggal_Akhir=("Tanggal", "last"),
        )
    )
    df_agregasi = df_agregasi.merge(df_tanggal, on="Kabupaten", how="left")

    for col in ["Harga_Rata2", "Harga_Min", "Harga_Max"]:
        df_agregasi[col] = pd.to_numeric(df_agregasi[col], errors="coerce").round(0).astype(int)
    for col in ["Harga_Std", "Harga_Skewness"]:
        df_agregasi[col] = pd.to_numeric(df_agregasi[col], errors="coerce").round(4)
    df_agregasi = df_agregasi.sort_values("Kabupaten").reset_index(drop=True)

    st.markdown("### Hasil Agregasi per Kabupaten/Kota")
    st.caption(
    f"Data harian sebanyak **{len(df_long_f):,} baris** berhasil diagregasi menjadi "
    f"**{len(df_agregasi):,} baris** (satu baris per kabupaten/kota)."
)
    st.dataframe(df_agregasi, use_container_width=True, hide_index=True)

    c_agg1, c_agg2 = st.columns(2)
    c_agg1.metric("Jumlah baris sebelum agregasi (harian)", f"{len(df_long_f):,}")
    c_agg2.metric("Jumlah baris sesudah agregasi (per kabupaten/kota)", f"{len(df_agregasi):,}")

    st.download_button(
    "Download Hasil Agregasi per Kabupaten/Kota (CSV)",
    data=df_to_csv_download(df_agregasi),
    file_name="data_agregasi_per_kabupaten.csv",
    mime="text/csv"
)

    # =========================================================
    # 4) TRANSFORMASI FITUR UNTUK K-MEANS
    # =========================================================
st.markdown("---")
with st.expander("4) Transformasi Fitur untuk K-Means", expanded=False):

    st.write(
    "Data harga harian tidak dapat langsung dipakai dalam K-Means, sehingga perlu diringkas menjadi "
    "fitur numerik per kabupaten/kota. Fitur clustering yang dipakai adalah lima atribut harga secara "
    "langsung: Harga_Rata2 (harga rata-rata), Harga_Min (harga minimum), Harga_Max (harga maksimum), "
    "Harga_Std (standar deviasi/fluktuasi harga), dan Harga_Skewness (kemiringan sebaran harga)."
)

    st.markdown("### Tabel Fitur yang Digunakan")
    st.dataframe(pd.DataFrame(FEATURE_DEFINITIONS), use_container_width=True, hide_index=True)

    kolom_deskriptif_tersedia = [c for c in DESCRIPTIVE_COLS if c in df_agregasi.columns]
    df_feat = df_agregasi[["Kabupaten"] + FEATURE_COLS + kolom_deskriptif_tersedia].copy()

    feature_cols = FEATURE_COLS


    st.markdown("### Dataset Fitur yang Dipakai Clustering")
    st.dataframe(df_feat, use_container_width=True, hide_index=True)

    if len(df_feat) < 3:
        st.error("Data kabupaten/kota terlalu sedikit untuk menentukan K optimal.")
        st.stop()

    # =========================================================
    # 5) STANDARDISASI DATA
    # =========================================================
st.markdown("---")
with st.expander("5) Standardisasi Data", expanded=False):

    X_before = df_feat[feature_cols].copy()
    for c in feature_cols:
        X_before[c] = pd.to_numeric(X_before[c], errors="coerce")

    X_before = X_before.dropna().copy()
    df_feat = df_feat.loc[X_before.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_before)

    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_feat.index)
    df_normalisasi = pd.concat(
        [df_feat[["Kabupaten"]].reset_index(drop=True), X_scaled_df.reset_index(drop=True)],
        axis=1,
    )

    coln1, coln2 = st.columns(2)
    with coln1:
        st.markdown("**Sebelum standardisasi**")
        st.dataframe(pd.concat([df_feat[["Kabupaten"]], X_before], axis=1), use_container_width=True, hide_index=True)
    with coln2:
        st.markdown("**Sesudah StandardScaler**")
        st.dataframe(df_normalisasi, use_container_width=True, hide_index=True)

    st.markdown("""
> **Penjelasan:**

Data distandardisasi menggunakan **StandardScaler**, bukan MinMaxScaler. Perubahan ini diterapkan
karena fitur K-Means tidak hanya berisi harga rata-rata, minimum, dan maksimum, tetapi juga
**standar deviasi** serta **skewness** yang mempunyai rentang dan karakter distribusi berbeda.
StandardScaler membuat setiap fitur memiliki rata-rata mendekati 0 dan standar deviasi mendekati 1,
sehingga tidak ada satu fitur yang mendominasi jarak Euclidean hanya karena skalanya lebih besar.

**Rumus StandardScaler (Z-score):**
""")

    st.latex(r"""
Z = \frac{X - \mu}{\sigma}
""")

    st.markdown(r"""
**Keterangan:**

- $Z$ adalah nilai setelah standardisasi.
- $X$ adalah nilai asli.
- $\mu$ adalah rata-rata fitur.
- $\sigma$ adalah standar deviasi populasi fitur.
""")

    contoh_fitur = feature_cols[0]
    contoh_x = float(X_before.iloc[0][contoh_fitur])
    contoh_mean = float(X_before[contoh_fitur].mean())
    contoh_std = float(X_before[contoh_fitur].std(ddof=0))
    contoh_hasil = (contoh_x - contoh_mean) / contoh_std if contoh_std > 0 else 0.0

    st.markdown(f"""
**Contoh Perhitungan Manual StandardScaler:**

Contoh pada fitur **{contoh_fitur}** untuk kabupaten/kota **{df_feat.iloc[0]['Kabupaten']}**:
""")

    st.latex(rf"""
Z = \frac{{{contoh_x:.2f} - {contoh_mean:.2f}}}{{{contoh_std:.2f}}} = {contoh_hasil:.4f}
""")

    st.caption(
        "Nilai negatif berarti berada di bawah rata-rata fitur, nilai positif berada di atas rata-rata, "
        "dan nilai sekitar 0 berada dekat rata-rata."
    )

    # =========================================================
    # 5) ELBOW METHOD UNTUK PEMILIHAN K (K = 2 s/d 10)
    # =========================================================
    k_min = 2
    k_max = min(9, len(df_feat) - 1)

st.markdown("---")
with st.expander(f"6) Penentuan Jumlah Cluster (K Kandidat + Evaluasi Silhouette) (K = {k_min} sampai {k_max})", expanded=False):

    if k_max < k_min:
        st.error("Jumlah kabupaten/kota terlalu sedikit untuk proses clustering (minimal 3 kabupaten/kota).")
        st.stop()

    ks = list(range(k_min, k_max + 1))

    inertias, silhouettes_per_k, ukuran_cluster_all, ukuran_cluster_min = cached_elbow_silhouette_search(
        X_scaled, tuple(ks)
    )

    # =========================================================
    # PEMILIHAN K — ELBOW MEMBENTUK BEBERAPA KANDIDAT, SILHOUETTE MENGEVALUASI KANDIDAT
    # =========================================================
    # Kandidat utama mengikuti arahan bimbingan: K=2, K=3, dan K=4.
    # Kurva WCSS tetap dihitung pada seluruh rentang K, tetapi tidak digunakan
    # untuk menampilkan satu titik siku otomatis yang dapat membingungkan.
    elbow_k = 3  # hanya nilai fallback internal; tidak ditampilkan sebagai keputusan
    kandidat_k = buat_kandidat_dari_elbow(elbow_k, ks, jumlah_minimal=3, total_objek=len(df_feat))
    if len(kandidat_k) < 3:
        st.warning(
            "Jumlah kabupaten/kota belum cukup untuk menguji lengkap K=2, K=3, dan K=4. "
            f"Kandidat valid yang dapat dihitung hanya: {', '.join(map(str, kandidat_k))}."
        )
    best_k, best_silhouette_k, best_silhouette_score, catatan_pemilihan_k = pilih_k_final_dari_kandidat(
        kandidat_k=kandidat_k,
        ks=ks,
        silhouettes_per_k=silhouettes_per_k,
        ukuran_cluster_all=ukuran_cluster_all,
        elbow_k=3,
        min_cluster_size=2,
        tie_tolerance=1e-9,
    )

    silhouette_final = float(silhouettes_per_k[ks.index(best_k)]) if best_k in ks else np.nan
    sil_final_text = f"{silhouette_final:.4f}" if np.isfinite(silhouette_final) else "Tidak dapat dihitung"
    kandidat_text = ", ".join([f"K = {k}" for k in kandidat_k])

    elbow_df = buat_tabel_validasi_k(
        ks=ks,
        inertias=inertias,
        silhouettes_per_k=silhouettes_per_k,
        ukuran_cluster_all=ukuran_cluster_all,
        kandidat_k=kandidat_k,
        elbow_k=elbow_k,
        best_k=best_k,
        best_silhouette_k=best_silhouette_k,
        min_cluster_size=2,
    )

    st.markdown("#### Hasil Penentuan K")
    c_k1, c_k2, c_k3 = st.columns(3)
    c_k1.metric("Rentang kurva Elbow/WCSS", f"K = {k_min}–{k_max}")
    c_k2.metric("Kandidat yang diperiksa", kandidat_text)
    c_k3.metric("K final", f"K = {best_k}")

    st.info(
        f"Kurva WCSS dihitung pada **K={k_min} sampai K={k_max}** untuk memperlihatkan pola Elbow. "
        f"Sesuai arahan bimbingan, kandidat yang benar-benar diperiksa adalah **{kandidat_text}**. "
        "Setiap kandidat terlebih dahulu dicek distribusi anggotanya; kandidat dengan cluster tunggal "
        "tidak menjadi pilihan utama. Dari kandidat yang distribusinya layak, K final ditentukan oleh "
        f"Silhouette Score tertinggi. Hasilnya adalah **K={best_k}** dengan Silhouette **{sil_final_text}**. "
        f"{catatan_pemilihan_k}"
    )

    st.success(
        f"**KEPUTUSAN K FINAL: K = {best_k}.** Keputusan ini berasal dari evaluasi K=2, K=3, dan K=4 "
        "berdasarkan dua hal yang diminta pembimbing: kualitas Silhouette dan distribusi anggota cluster."
    )

    # =========================================================
    # GRAFIK 1 — ELBOW METHOD (WCSS)
    # =========================================================
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(ks, inertias, marker="o", linewidth=1.5, color="#4C72B0", zorder=2)
    if kandidat_k:
        ax.axvspan(
            min(kandidat_k) - 0.08,
            max(kandidat_k) + 0.08,
            color="gray",
            alpha=0.14,
            label=f"Kandidat bimbingan: {kandidat_text}",
        )
    for k in kandidat_k:
        ax.axvline(k, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.scatter(
        [best_k],
        [inertias[ks.index(best_k)]],
        color="red",
        s=105,
        zorder=4,
        marker="D",
        label=f"K final = {best_k}",
    )
    ax.set_title(f"Grafik Elbow/WCSS K={k_min}–{k_max}; validasi kandidat K=2, K=3, K=4", fontsize=9)
    ax.set_xlabel("Jumlah Cluster (K)", fontsize=9)
    ax.set_ylabel("WCSS", fontsize=9)
    ax.set_xticks(ks)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Kurva WCSS ditampilkan tanpa memaksakan satu titik siku otomatis. "
        "Kurva K=2–9 tetap ditampilkan sebagai bukti pola WCSS, sedangkan area K=2–4 adalah kandidat yang "
        "dibandingkan menggunakan Silhouette dan distribusi anggota."
    )

    st.markdown("### Tabel Evaluasi Semua K")
    st.caption(
        f"Tabel ini menampilkan seluruh rentang K = {k_min} sampai {k_max}. K=2, K=3, dan K=4 menjadi kandidat utama; kandidat harus memiliki distribusi layak sebelum dibandingkan berdasarkan Silhouette."
    )
    st.dataframe(elbow_df, use_container_width=True, hide_index=True)

    kandidat_df = elbow_df[elbow_df["K"].isin(kandidat_k)].copy()
    kandidat_df["Peringkat_Silhouette_di_Kandidat"] = (
        kandidat_df["Silhouette_Score"].rank(ascending=False, method="min")
        .apply(lambda x: int(x) if pd.notna(x) else "Tidak valid")
    )
    kandidat_df = kandidat_df.sort_values(
        ["K_Dipakai", "Silhouette_Score", "Rasio_Kepadatan", "K"],
        ascending=[False, False, False, True]
    )

    st.markdown("### Tabel Validasi Kandidat K = 2, 3, dan 4")
    st.caption(
        "Tabel ini mengikuti alur validasi pembimbing: K=2, K=3, dan K=4 dibandingkan memakai Silhouette, "
        "jumlah anggota cluster, minimum anggota, maksimum anggota, dan rasio kepadatan. Distribusi anggota "
        "menjadi syarat kelayakan; setelah itu Silhouette Score menentukan K final."
    )
    st.dataframe(kandidat_df, use_container_width=True, hide_index=True)

    # Perbandingan khusus sesuai permintaan penguji 1: coba K=2 dan K=3.
    kandidat_23 = kandidat_df[kandidat_df["K"].isin([2, 3])].copy().sort_values("K")
    st.markdown("### Perbandingan Khusus K = 2 dan K = 3")
    st.caption(
        "Bagian ini menjawab arahan penguji untuk mencoba K=2 dan K=3. Distribusi anggota "
        "diperiksa terlebih dahulu. Di antara kandidat yang distribusinya layak, nilai Silhouette "
        "yang lebih tinggi menunjukkan pemisahan cluster yang lebih baik."
    )
    st.dataframe(
        kandidat_23[[
            "K", "Silhouette_Score", "Kualitas_Silhouette", "Ukuran_Cluster",
            "Min_Anggota", "Max_Anggota", "Rasio_Kepadatan", "Distribusi_Layak"
        ]],
        use_container_width=True,
        hide_index=True,
    )
    kandidat_23_layak = kandidat_23[kandidat_23["Distribusi_Layak"] == "Ya"].copy()
    if not kandidat_23_layak.empty:
        kandidat_23_layak = kandidat_23_layak.sort_values(
            ["Silhouette_Score", "Rasio_Kepadatan", "K"],
            ascending=[False, False, True],
        )
        k_terbaik_23 = int(kandidat_23_layak.iloc[0]["K"])
        sil_terbaik_23 = float(kandidat_23_layak.iloc[0]["Silhouette_Score"])
        st.info(
            f"Hasil perbandingan khusus K=2 dan K=3: **K={k_terbaik_23}** menjadi pilihan "
            f"di antara kandidat yang distribusinya layak, dengan Silhouette **{sil_terbaik_23:.4f}**."
        )
    else:
        st.warning(
            "K=2 dan K=3 sama-sama memiliki distribusi yang tidak layak karena terdapat cluster tunggal. "
            "Keduanya tetap ditampilkan, tetapi keputusan harus diberi catatan keterbatasan."
        )

    distribusi_rows = []
    for k in kandidat_k:
        idx_k = ks.index(int(k))
        for nomor_cluster, jumlah_anggota in enumerate(ukuran_cluster_all[idx_k], start=1):
            distribusi_rows.append({
                "K": int(k),
                "Cluster": f"C{nomor_cluster}",
                "Jumlah_Anggota": int(jumlah_anggota),
                "Status_K": "K final" if int(k) == int(best_k) else "Kandidat pembanding",
            })
    st.markdown("### Distribusi Anggota Setiap Cluster pada K Kandidat")
    st.caption(
        "Jumlah anggota ditampilkan per cluster dan tidak diurutkan dari angka terkecil ke terbesar. "
        "C1 sampai CK mengikuti urutan centroid Harga_Rata2 dari rendah ke tinggi, sehingga label cluster "
        "konsisten dengan tabel anggota dan interpretasi kategori."
    )
    st.dataframe(pd.DataFrame(distribusi_rows), use_container_width=True, hide_index=True)

    st.markdown("### Ringkasan Validasi Distribusi Anggota")
    min_final = int(kandidat_df.loc[kandidat_df["K"] == best_k, "Min_Anggota"].iloc[0]) if (kandidat_df["K"] == best_k).any() else 0
    rasio_final = float(kandidat_df.loc[kandidat_df["K"] == best_k, "Rasio_Kepadatan"].iloc[0]) if (kandidat_df["K"] == best_k).any() else 0.0
    distribusi_final_layak = min_final >= 2
    if distribusi_final_layak:
        st.success(
            f"K final = **{best_k}** memiliki minimal anggota cluster **{min_final}** dan rasio "
            f"kepadatan **{rasio_final:.3f}**. Tidak ada cluster tunggal, sehingga distribusinya "
            "layak untuk diinterpretasikan."
        )
    else:
        st.warning(
            f"K final = **{best_k}** memiliki minimal anggota cluster **{min_final}**. Semua kandidat "
            "yang tersedia mempunyai masalah distribusi, sehingga hasil ini harus diberi catatan keterbatasan."
        )

    # =========================================================
    # PENJELASAN PEMILIHAN K + SILHOUETTE PLOT SETIAP KANDIDAT
    # =========================================================
    st.markdown(f"""
### Penjelasan Pemilihan K

1. Kurva **Elbow/WCSS** dihitung pada K={k_min} sampai K={k_max} untuk memperlihatkan perubahan WCSS.
2. Sesuai arahan bimbingan, kandidat yang diperiksa adalah **{kandidat_text}**.
3. Setiap kandidat diperiksa distribusinya. Kandidat dengan cluster beranggotakan satu tidak menjadi pilihan utama.
4. Dari kandidat yang distribusinya layak, dipilih nilai **Silhouette Score tertinggi**.
5. Distribusi anggota, nama anggota, serta nilai minimum, rata-rata, dan maksimum Silhouette ditampilkan agar asal angkanya jelas.

Hasil perhitungan menetapkan **K final = {best_k}** dengan Silhouette **{sil_final_text}**.
""")

    st.markdown("#### Ringkasan Kandidat K = 2, 3, dan 4")
    ringkasan_kandidat_visual = kandidat_df[[
        "K", "Silhouette_Score", "Kualitas_Silhouette", "Ukuran_Cluster",
        "Min_Anggota", "Max_Anggota", "Rasio_Kepadatan", "Distribusi_Layak",
        "K_Dipakai", "Catatan_Pemilihan"
    ]].copy()
    st.dataframe(ringkasan_kandidat_visual, use_container_width=True, hide_index=True)

    for nomor_grafik, k in enumerate(kandidat_k, start=1):
        evaluasi_k = evaluasi_kandidat_lengkap(
            X_scaled=X_scaled,
            nama_objek=df_feat["Kabupaten"].reset_index(drop=True),
            k=int(k),
        )
        labels_k = evaluasi_k["labels"]
        sil_k = evaluasi_k["silhouette_avg"]
        counts_k = evaluasi_k["counts"]
        ringkasan_k = evaluasi_k["ringkasan"].copy()
        detail_k = evaluasi_k["detail"].copy()
        bukti_k = evaluasi_k["bukti"].copy()
        rumus_total_k = evaluasi_k["rumus_total"]

        # Kembalikan centroid dari skala StandardScaler ke satuan harga asli.
        # C1 selalu centroid Harga_Rata2 terendah, lalu C2, C3, dan seterusnya.
        centroids_original_k = scaler.inverse_transform(evaluasi_k["centroids"])
        centroid_harga_k = centroids_original_k[:, feature_cols.index("Harga_Rata2")]
        ringkasan_k.insert(
            2,
            "Centroid_Harga_Rata2",
            [round(float(v), 2) for v in centroid_harga_k],
        )
        ringkasan_k.insert(
            3,
            "Dasar_Penamaan_Kategori",
            [
                f"Urutan centroid Harga_Rata2 ke-{i + 1} dari rendah ke tinggi"
                for i in range(len(ringkasan_k))
            ],
        )
        bukti_k = bukti_k.merge(
            ringkasan_k[["Cluster", "Centroid_Harga_Rata2", "Dasar_Penamaan_Kategori"]],
            on="Cluster",
            how="left",
        )

        min_k = min(counts_k) if counts_k else 0
        max_k = max(counts_k) if counts_k else 0
        rasio_k = rasio_kepadatan_cluster(counts_k)
        distribusi_k_layak = min_k >= 2

        if int(k) == int(best_k):
            status_k = "K final"
        elif best_silhouette_k is not None and int(k) == int(best_silhouette_k):
            status_k = "Silhouette tertinggi secara murni"
        else:
            status_k = "kandidat pembanding"

        st.markdown(
            f"**Grafik 2.{nomor_grafik} — K = {k} ({status_k})**  \n"
            f"Silhouette rata-rata: **{sil_k:.4f}** | "
            f"Ukuran cluster: **{format_ukuran_cluster(counts_k)}** | "
            f"Min anggota: **{min_k}** | Max anggota: **{max_k}** | "
            f"Rasio kepadatan: **{rasio_k:.3f}** | "
            f"Distribusi layak: **{'Ya' if distribusi_k_layak else 'Tidak'}**"
        )
        st.caption(
            "C1 adalah cluster dengan centroid Harga_Rata2 terendah, kemudian C2, C3, dan seterusnya. "
            "Nomor cluster tidak mengikuti label acak bawaan K-Means."
        )
        if not distribusi_k_layak:
            st.warning(
                f"K={k} memiliki cluster dengan hanya {min_k} anggota. Kandidat ini tetap ditampilkan "
                "untuk perbandingan, tetapi tidak menjadi pilihan utama selama ada kandidat lain yang distribusinya layak."
            )

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        plot_silhouette(ax, X_scaled, labels_k, int(k))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"##### Audit lengkap asal angka pada Grafik 2.{nomor_grafik}")
        st.markdown(
            "- **C1, C2, dan seterusnya** berasal dari urutan centroid `Harga_Rata2`, bukan label acak K-Means.  \n"
            "- **n** berasal dari jumlah nama kabupaten/kota yang masuk ke cluster tersebut.  \n"
            "- **mean** berasal dari jumlah seluruh nilai Silhouette anggota dibagi `n`.  \n"
            "- **min** dan **max** adalah nilai Silhouette terkecil dan terbesar beserta nama pemilik nilainya.  \n"
            "- **garis merah rata-rata total** dihitung dari seluruh nilai anggota, atau setara dengan rata-rata tertimbang mean setiap cluster."
        )

        st.info(
            "Jangan tertukar: **Min_Anggota/Max_Anggota** pada tabel validasi K adalah jumlah anggota "
            "cluster terkecil/terbesar. Sementara **min/max pada grafik Silhouette** adalah nilai "
            "Silhouette terendah/tertinggi di dalam satu cluster."
        )

        st.markdown("**A. Dasar nomor cluster, kategori, jumlah, dan anggota**")
        st.dataframe(ringkasan_k, use_container_width=True, hide_index=True)

        st.markdown("**B. Bukti perhitungan n, mean, min, dan max**")
        st.dataframe(bukti_k, use_container_width=True, hide_index=True)

        st.markdown("**C. Perhitungan garis merah rata-rata total**")
        st.code(rumus_total_k, language=None)
        st.caption(
            "Label mean pada sumbu Y dibulatkan 3 desimal, sedangkan garis rata-rata total "
            "ditampilkan 4 desimal. Perbedaan kecil pada angka tampilan hanya berasal dari pembulatan."
        )

        st.markdown(f"**D. Detail nilai setiap kabupaten/kota — K = {k}**")
        st.dataframe(detail_k, use_container_width=True, hide_index=True)

        total_n_k = int(ringkasan_k["Jumlah_Anggota_n"].sum()) if not ringkasan_k.empty else 0
        selisih_rata_k = abs(
            float(evaluasi_k["silhouette_avg"]) -
            float(evaluasi_k["silhouette_avg_tertimbang"])
        ) if np.isfinite(evaluasi_k["silhouette_avg_tertimbang"]) else np.nan
        st.success(
            f"Verifikasi otomatis: total n = **{total_n_k}** sama dengan jumlah objek = **{len(df_feat)}**. "
            f"Rata-rata langsung `silhouette_samples` = **{sil_k:.6f}** dan rata-rata tertimbang cluster = "
            f"**{evaluasi_k['silhouette_avg_tertimbang']:.6f}** "
            f"(selisih = **{selisih_rata_k:.10f}**)."
        )

    st.markdown("### Ringkasan Jawaban untuk Pertanyaan Dosen Pembimbing")
    st.markdown(f"""
- **K=2 sampai K={k_max}** adalah rentang perhitungan WCSS, bukan jumlah cluster final.
- **K=2, K=3, dan K=4** adalah kandidat yang dibandingkan sesuai arahan bimbingan.
- **K final = {best_k}** karena memiliki Silhouette terbaik di antara kandidat yang distribusinya layak.
- **C1, C2, dan seterusnya** disusun dari centroid Harga_Rata2 terendah ke tertinggi.
- **n** adalah jumlah nama anggota; **mean/min/max** berasal dari nilai Silhouette anggota cluster.
- **Rendah/Sedang/Tinggi** ditentukan oleh centroid harga, bukan jumlah anggota.
- Data **outlier** ditampilkan agar dapat diverifikasi apakah kondisi riil atau salah pencatatan.
""")

    st.markdown(f"""
### Kesimpulan Pemilihan K

K final **bukan dipilih dari seluruh K=2–9 hanya karena WCSS terus menurun**. Rentang tersebut digunakan
untuk menampilkan kurva Elbow. Sesuai arahan bimbingan, keputusan difokuskan pada **K=2, K=3, dan K=4**.
Kandidat dengan cluster tunggal dikeluarkan dari pilihan utama; dari kandidat yang distribusinya layak,
dipilih Silhouette Score tertinggi. Berdasarkan aturan tersebut, hasil akhirnya adalah **K={best_k}**
dengan Silhouette **{sil_final_text}**.

**Catatan keputusan:** {catatan_pemilihan_k}
""")

    if st.checkbox("📖 Tampilkan rumus Elbow, Silhouette, dan distribusi anggota", key="detail_k_optimal"):
        st.markdown(r"""
### Rumus WCSS pada Elbow Method

$$
WCSS=\sum_{i=1}^{k}\sum_{x \in C_i}||x-\mu_i||^2
$$

WCSS adalah jumlah kuadrat jarak setiap data terhadap centroid cluster. Kurva WCSS digunakan untuk
melihat berkurangnya variasi dalam cluster ketika K bertambah. Dalam analisis ini, kurva dihitung pada
K=2 sampai K maksimum, sedangkan kandidat keputusan mengikuti arahan bimbingan: K=2, K=3, dan K=4.

### Rumus Silhouette Score

$$
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
$$

- $a(i)$ = rata-rata jarak objek ke anggota cluster yang sama.
- $b(i)$ = rata-rata jarak objek ke cluster tetangga terdekat.
- Nilai mendekati 1 menunjukkan pemisahan yang lebih baik.

### Pemeriksaan Distribusi Anggota

Kandidat dinyatakan layak apabila setiap cluster memiliki minimal dua anggota. Ketentuan ini mencegah
cluster tunggal dipilih sebagai hasil utama karena nilai Silhouette anggota tunggal tidak memiliki
pembanding internal yang memadai.

$$
Rasio\ Kepadatan=\frac{Anggota\ cluster\ terkecil}{Anggota\ cluster\ terbesar}
$$

Rasio kepadatan digunakan sebagai informasi keseimbangan dan sebagai pemecah seri jika nilai
Silhouette sama; bukan untuk mengalahkan perbedaan Silhouette yang jelas di antara kandidat layak.
""")

        tabel_perhitungan_wcss = pd.DataFrame({
            "K": ks,
            "WCSS": [round(float(v), 6) for v in inertias],
            "Kandidat_K_2_3_4": ["Ya" if k in kandidat_k else "Tidak" for k in ks],
            "Silhouette_Score": [
                np.nan if pd.isna(v) else round(float(v), 6) for v in silhouettes_per_k
            ],
            "Distribusi_Anggota": [format_ukuran_cluster(v) for v in ukuran_cluster_all],
            "Distribusi_Layak": ["Ya" if min(v) >= 2 else "Tidak" for v in ukuran_cluster_all],
            "K_Final": ["Ya" if int(k) == int(best_k) else "Tidak" for k in ks],
        })
        st.markdown("### Angka Perhitungan pada Data yang Diunggah")
        st.dataframe(tabel_perhitungan_wcss, use_container_width=True, hide_index=True)

    # =========================================================
    # 6) K-MEANS FINAL DAN EVALUASI
    # =========================================================
st.markdown("---")
with st.expander("7) Proses Clustering", expanded=False):

    labels_final_old, centroids_scaled_old, _, model_kmeans_final = jalankan_kmeans_sklearn(X_scaled, int(best_k))

    df_result = df_feat.copy()
    df_result["Cluster_Lama"] = labels_final_old

    centroids_original_old = scaler.inverse_transform(centroids_scaled_old)
    df_centroid = pd.DataFrame(centroids_original_old, columns=feature_cols)
    df_centroid["Cluster_Lama"] = range(len(df_centroid))

    urutan_cluster = df_centroid.sort_values("Harga_Rata2").reset_index(drop=True)
    label_kategori = buat_label_kategori(len(urutan_cluster))

    mapping_cluster = {}
    kategori_map = {}

    for idx, row in urutan_cluster.iterrows():
        old_cluster = int(row["Cluster_Lama"])
        new_cluster = idx + 1
        mapping_cluster[old_cluster] = new_cluster
        kategori_map[new_cluster] = label_kategori[idx]

    df_result["Cluster"] = df_result["Cluster_Lama"].map(mapping_cluster)
    df_result["Kategori"] = df_result["Cluster"].map(kategori_map)

    jarak_ke_centroid = []
    for i, old_label in enumerate(labels_final_old):
        jarak = np.linalg.norm(X_scaled[i] - centroids_scaled_old[old_label])
        jarak_ke_centroid.append(jarak)

    df_result["Jarak_Euclidean_ke_Centroid"] = np.round(jarak_ke_centroid, 4)
    df_result = df_result.drop(columns=["Cluster_Lama"])

    df_centroid["Cluster"] = df_centroid["Cluster_Lama"].map(mapping_cluster)
    df_centroid["Kategori"] = df_centroid["Cluster"].map(kategori_map)
    df_centroid = df_centroid.drop(columns=["Cluster_Lama"])
    df_centroid = df_centroid.sort_values("Cluster").reset_index(drop=True)

    st.markdown("### Tahapan Proses Clustering K-Means")
    st.info(
        "Proses clustering pada aplikasi ini menggunakan class `KMeans` dari scikit-learn "
        "dengan `init='k-means++'`, `n_init=10`, `max_iter=300`, dan `random_state=42`."
    )
    st.markdown(r"""
Setelah titik siku Elbow diketahui dan kandidat K=2, K=3, serta K=4 dievaluasi menggunakan Silhouette Score, proses clustering dilakukan dengan tahapan berikut:

1. Menginisialisasi nilai K.
2. K-Means scikit-learn memilih centroid awal dengan `init="k-means++"`.
3. Menghitung jarak Euclidean antara data dan centroid.
4. Mengelompokkan data ke centroid terdekat.
5. Menghitung ulang centroid sampai hasil stabil.
6. Menghasilkan label cluster untuk setiap data.

Rumus Euclidean Distance:

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

Keterangan:
- $x_i$ adalah nilai fitur data.
- $y_i$ adalah nilai fitur centroid.
- $d(x,y)$ adalah jarak antara data dan centroid.
""")

    contoh_idx = 0
    contoh_data = X_scaled[contoh_idx]
    contoh_label = labels_final_old[contoh_idx]
    contoh_centroid = centroids_scaled_old[contoh_label]
    contoh_selisih_kuadrat = (contoh_data - contoh_centroid) ** 2
    contoh_jarak = np.sqrt(contoh_selisih_kuadrat.sum())

    fitur_tampil = feature_cols

    baris_data = []
    baris_centroid = []
    baris_selisih = []
    rumus_terms = []

    for idx_f, nama_fitur in enumerate(fitur_tampil):
        baris_data.append(f"- {nama_fitur} = {contoh_data[idx_f]:.4f}")
        baris_centroid.append(f"- {nama_fitur} = {contoh_centroid[idx_f]:.4f}")
        baris_selisih.append(f"- ({contoh_data[idx_f]:.4f} - {contoh_centroid[idx_f]:.4f})² = {contoh_selisih_kuadrat[idx_f]:.6f}")
        rumus_terms.append(f"({contoh_data[idx_f]:.4f}-{contoh_centroid[idx_f]:.4f})^2")

    st.markdown(f"""
### Contoh Perhitungan Euclidean Distance

Contoh data: **{df_feat.iloc[contoh_idx]['Kabupaten']}**

Data hasil standardisasi:

{chr(10).join(baris_data)}

Centroid cluster terdekat:

{chr(10).join(baris_centroid)}

Selisih kuadrat:

{chr(10).join(baris_selisih)}

Rumus:

$$
d = {chr(92)}sqrt{{ {' + '.join(rumus_terms)} }}
$$

Hasil perhitungan jarak seluruh fitur:

$$
d = {contoh_jarak:.4f}
$$

Nilai jarak ini menunjukkan kedekatan data terhadap centroid cluster-nya. Semakin kecil jarak, semakin dekat data tersebut dengan centroid.
""")

    st.markdown("### Jarak Data ke Centroid Terdekat")
    st.dataframe(
    df_result[["Kabupaten", "Cluster", "Kategori", "Jarak_Euclidean_ke_Centroid"]].sort_values(["Cluster", "Kabupaten"]),
    use_container_width=True,
    hide_index=True
)

    st.write(f"Nilai K final yang digunakan: **{best_k}**")
    st.write("Hasil clustering:")
    st.dataframe(df_result.sort_values(["Cluster", "Kabupaten"]), use_container_width=True, hide_index=True)

    st.markdown("### Centroid Akhir Tiap Cluster")
    for c in feature_cols:
        df_centroid[c] = df_centroid[c].round(2)
    st.dataframe(df_centroid, use_container_width=True, hide_index=True)

    st.markdown("""
**Penjelasan Centroid:**

Centroid adalah titik pusat dari setiap cluster.
Setiap kabupaten/kota masuk ke cluster dengan centroid terdekat berdasarkan **jarak Euclidean** pada data hasil standardisasi.
Bagian ini ditampilkan agar asal pengelompokan kabupaten/kota dapat dijelaskan.
""")

    st.markdown("### Anggota Tiap Cluster (Hasil Akhir: Harga Gula Pasir per Daerah)")
    for i in sorted(df_result["Cluster"].unique()):
        anggota = df_result[df_result["Cluster"] == i]["Kabupaten"].astype(str).unique().tolist()
        kategori = kategori_map.get(i, "")
        st.write(f"**Cluster {i} ({kategori})**: {', '.join(anggota)}")

    # =========================================================
    # 7) EVALUASI HASIL CLUSTERING
    # =========================================================
st.markdown("---")
with st.expander("8) Evaluasi Hasil Clustering", expanded=False):
    silhouette_avg = hitung_silhouette_aman(X_scaled, labels_final_old)

    if not np.isfinite(silhouette_avg):
        st.error(
            "Silhouette Score tidak dapat dihitung karena hasil K-Means hanya membentuk satu label cluster efektif "
            "atau jumlah label tidak memenuhi syarat evaluasi. Coba gunakan data yang lebih bervariasi atau turunkan rentang K."
        )
        st.stop()

    st.markdown("#### Tabel Interpretasi Nilai Silhouette Score")

    tabel_interpretasi_sil = pd.DataFrame({
    "Nilai Silhouette": ["0.71 - 1.00", "0.51 - 0.70", "0.26 - 0.50", "≤ 0.25"],
    "Interpretasi": [
        "Struktur yang dihasilkan kuat",
        "Struktur yang dihasilkan baik",
        "Struktur yang dihasilkan lemah",
        "Tidak terstruktur"
    ]
})

    st.dataframe(tabel_interpretasi_sil, use_container_width=True, hide_index=True)

    st.metric("Silhouette Score (Rata-rata Keseluruhan)", f"{silhouette_avg:.4f}")

    ket_sil = label_kategori_silhouette(silhouette_avg)

    if silhouette_avg >= 0.71:
        st.success(
        f"Nilai Silhouette Score sebesar **{silhouette_avg:.4f}** termasuk dalam rentang **0.71 – 1.00**, "
        f"sehingga struktur cluster yang dihasilkan tergolong **kuat**."
    )
    elif silhouette_avg >= 0.51:
        st.info(
        f"Nilai Silhouette Score sebesar **{silhouette_avg:.4f}** termasuk dalam rentang **0.51 – 0.70**, "
        f"sehingga struktur cluster yang dihasilkan tergolong **baik**."
    )
    elif silhouette_avg >= 0.26:
        st.warning(
        f"Nilai Silhouette Score sebesar **{silhouette_avg:.4f}** termasuk dalam rentang **0.26 – 0.50**, "
        f"sehingga struktur cluster yang dihasilkan tergolong **lemah**."
    )
    else:
        st.error(
        f"Nilai Silhouette Score sebesar **{silhouette_avg:.4f}** termasuk dalam rentang **≤ 0.25**, "
        f"sehingga struktur cluster yang dihasilkan **tidak terstruktur**."
    )
    st.markdown("""
> Interpretasi nilai Silhouette Score:

- Nilai mendekati **1** berarti cluster terbentuk dengan baik.
- Nilai mendekati **0** berarti jarak antar cluster saling berdekatan atau tumpang tindih.
- Nilai negatif berarti data kemungkinan masuk ke cluster yang kurang tepat.
""")

    # -----------------------------------------------------------
    # DETAIL SILHOUETTE PER DATA
    # -----------------------------------------------------------
    sample_silhouette_values = silhouette_samples(X_scaled, labels_final_old)

    if len(sample_silhouette_values) == len(df_result):

        df_silhouette = pd.DataFrame({
        "Kabupaten": df_result["Kabupaten"].values,
        "Cluster": df_result["Cluster"].values,
        "Kategori Cluster": df_result["Kategori"].values,
        "Silhouette Score": np.round(sample_silhouette_values, 4)
    })

        df_silhouette["Kualitas"] = df_silhouette["Silhouette Score"].apply(kategori_silhouette)

        st.markdown("### 📊 Detail Silhouette Score per Kabupaten/Kota")
        st.dataframe(
        df_silhouette.sort_values("Silhouette Score", ascending=False),
        use_container_width=True,
        hide_index=True
    )

        # ---- RATA-RATA SILHOUETTE PER CLUSTER ----
        df_sil_per_cluster = (
        df_silhouette.groupby(["Cluster", "Kategori Cluster"])["Silhouette Score"]
        .agg(Rata2_Silhouette="mean", Jumlah_Anggota="count")
        .reset_index()
        .sort_values("Cluster")
    )
        df_sil_per_cluster["Rata2_Silhouette"] = df_sil_per_cluster["Rata2_Silhouette"].round(4)

        st.markdown("### 📌 Rata-rata Silhouette Score per Cluster")
        st.dataframe(df_sil_per_cluster, use_container_width=True, hide_index=True)

        cluster_lemah = df_sil_per_cluster[df_sil_per_cluster["Rata2_Silhouette"] < silhouette_avg]
        if len(cluster_lemah) > 0:
            daftar_cluster_lemah = ", ".join(
            f"Cluster {int(c)} ({k})" for c, k in zip(cluster_lemah["Cluster"], cluster_lemah["Kategori Cluster"])
        )
            st.warning(
            f"Cluster berikut memiliki rata-rata Silhouette **di bawah** rata-rata keseluruhan "
            f"({silhouette_avg:.4f}): **{daftar_cluster_lemah}**. Cluster ini paling perlu dicermati "
            f"karena anggotanya relatif kurang kompak dibanding cluster lain."
        )
        else:
            st.success("Seluruh cluster memiliki rata-rata Silhouette di atas atau setara rata-rata keseluruhan.")

        sil_tertinggi = df_silhouette.loc[df_silhouette["Silhouette Score"].idxmax()]
        sil_terendah  = df_silhouette.loc[df_silhouette["Silhouette Score"].idxmin()]

        st.markdown("### 📌 Ringkasan")

        st.success(
        f"Terbaik: {sil_tertinggi['Kabupaten']} "
        f"({sil_tertinggi['Silhouette Score']:.4f})"
    )

        st.error(
        f"Terendah: {sil_terendah['Kabupaten']} "
        f"({sil_terendah['Silhouette Score']:.4f})"
    )

        st.markdown(f"""
### 📄 Interpretasi Detail Silhouette

Nilai rata-rata Silhouette sebesar **{silhouette_avg:.4f}** menunjukkan kualitas clustering secara keseluruhan
dan tergolong **{ket_sil}** berdasarkan tabel interpretasi di atas.

Kabupaten/kota dengan nilai tertinggi adalah **{sil_tertinggi['Kabupaten']}** ({sil_tertinggi['Silhouette Score']:.4f}),
yang menunjukkan data tersebut paling sesuai dengan cluster-nya.

Kabupaten/kota dengan nilai terendah adalah **{sil_terendah['Kabupaten']}** ({sil_terendah['Silhouette Score']:.4f}),
yang menunjukkan adanya kedekatan dengan cluster lain.
""")

        # -----------------------------------------------------------
        # STATISTIK LENGKAP NILAI SILHOUETTE (SELURUH DATA)
        # -----------------------------------------------------------
        st.markdown("### 📈 Statistik Lengkap Nilai Silhouette")
        st.caption(
            "Statistik ini melengkapi nilai tertinggi/terendah di atas dengan gambaran sebaran seluruh nilai "
            "Silhouette, termasuk berapa banyak data yang bernilai negatif (indikasi data berada di area tumpang tindih antar cluster)."
        )

        n_data_sil = len(sample_silhouette_values)
        n_negatif = int(np.sum(sample_silhouette_values < 0))
        pct_negatif = (n_negatif / n_data_sil * 100) if n_data_sil > 0 else 0.0

        stat_sil = pd.DataFrame({
            "Statistik": [
                "Rata-rata", "Median", "Std. Deviasi", "Nilai Minimum", "Nilai Maksimum",
                "Jumlah Data Bernilai Negatif", "Persentase Data Bernilai Negatif",
            ],
            "Nilai": [
                f"{np.mean(sample_silhouette_values):.4f}",
                f"{np.median(sample_silhouette_values):.4f}",
                f"{np.std(sample_silhouette_values):.4f}",
                f"{np.min(sample_silhouette_values):.4f}",
                f"{np.max(sample_silhouette_values):.4f}",
                f"{n_negatif} dari {n_data_sil} data",
                f"{pct_negatif:.1f}%",
            ],
        })
        st.dataframe(stat_sil, use_container_width=True, hide_index=True)

        if n_negatif == 0:
            st.success(
                "Tidak ditemukan data dengan nilai Silhouette negatif. Artinya seluruh kabupaten/kota sudah "
                "ditempatkan pada cluster yang jaraknya lebih dekat dibandingkan cluster lain."
            )
        else:
            df_negatif = df_silhouette[df_silhouette["Silhouette Score"] < 0].sort_values("Silhouette Score")
            st.warning(
                f"Terdapat **{n_negatif} dari {n_data_sil} data ({pct_negatif:.1f}%)** dengan nilai Silhouette "
                "negatif. Artinya kabupaten/kota tersebut secara jarak fitur sebenarnya sedikit lebih dekat ke "
                "cluster lain dibandingkan cluster tempatnya berada saat ini, sehingga posisinya berada di area "
                "tumpang tindih (overlap) antar cluster."
            )
            st.dataframe(df_negatif, use_container_width=True, hide_index=True)

        # -----------------------------------------------------------
        # PERBANDINGAN SILHOUETTE K FINAL DENGAN K KANDIDAT LAIN
        # -----------------------------------------------------------
        st.markdown("### 🔁 Perbandingan Silhouette Score K Final dengan Kandidat K Lain")
        st.caption(
            "Tabel ini membandingkan Silhouette Score K final pada tahap ini dengan Silhouette Score kandidat K "
            "lain yang diuji pada bagian '6) Penentuan Jumlah Cluster'. Perbandingan ini menjadi bukti kuantitatif "
            "mengapa K final yang dipilih, bukan kandidat K yang lain."
        )

        kandidat_lebih_tinggi = pd.DataFrame()
        kandidat_layak_lebih_tinggi = pd.DataFrame()
        try:
            tabel_banding_k = kandidat_df[[
                "K", "Silhouette_Score", "Kualitas_Silhouette", "Min_Anggota",
                "Max_Anggota", "Rasio_Kepadatan", "Distribusi_Layak", "K_Dipakai",
                "Catatan_Pemilihan",
            ]].sort_values("K").reset_index(drop=True)
            st.dataframe(tabel_banding_k, use_container_width=True, hide_index=True)

            kandidat_lebih_tinggi = tabel_banding_k[
                (tabel_banding_k["K"] != int(best_k))
                & (tabel_banding_k["Silhouette_Score"] > float(silhouette_avg))
            ]
            kandidat_layak_lebih_tinggi = kandidat_lebih_tinggi[
                kandidat_lebih_tinggi["Distribusi_Layak"] == "Ya"
            ]

            if len(kandidat_layak_lebih_tinggi) > 0:
                daftar_k = ", ".join(
                    f"K={int(r['K'])} ({r['Silhouette_Score']:.4f})"
                    for _, r in kandidat_layak_lebih_tinggi.iterrows()
                )
                st.error(
                    f"Ditemukan kandidat **layak** dengan Silhouette lebih tinggi: {daftar_k}. "
                    "Kondisi ini menunjukkan ketidaksesuaian pemilihan K; bersihkan cache dan jalankan ulang."
                )
            elif len(kandidat_lebih_tinggi) > 0:
                daftar_k = ", ".join(
                    f"K={int(r['K'])} ({r['Silhouette_Score']:.4f}; distribusi tidak layak)"
                    for _, r in kandidat_lebih_tinggi.iterrows()
                )
                st.info(
                    f"Ada kandidat dengan Silhouette lebih tinggi secara murni, yaitu **{daftar_k}**, tetapi "
                    "kandidat tersebut tidak dipilih karena mempunyai cluster tunggal. K final merupakan "
                    "Silhouette tertinggi di antara kandidat yang distribusinya layak."
                )
            else:
                st.success(
                    f"K final **K={int(best_k)}** memiliki Silhouette tertinggi di antara kandidat "
                    "K=2, K=3, dan K=4 yang distribusinya layak."
                )
        except Exception:
            st.info("Tabel perbandingan kandidat K tidak tersedia untuk ditampilkan pada sesi ini.")

        # -----------------------------------------------------------
        # KESIMPULAN AKHIR EVALUASI CLUSTERING
        # -----------------------------------------------------------
        teks_status_tertinggi = (
            "merupakan kandidat dengan Silhouette Score tertinggi di antara kandidat yang distribusinya layak"
            if len(kandidat_layak_lebih_tinggi) == 0
            else "perlu dihitung ulang karena ditemukan kandidat layak dengan Silhouette lebih tinggi"
        )
        teks_reliabilitas = (
            "dapat digunakan sebagai dasar pengelompokan yang cukup dapat diandalkan"
            if silhouette_avg >= 0.26
            else "perlu dibaca sebagai eksplorasi pola awal, karena struktur pemisahan antar cluster masih tergolong lemah"
        )

        st.markdown("### ✅ Kesimpulan Evaluasi Clustering")
        st.markdown(f"""
Berdasarkan hasil evaluasi pada bagian ini, dapat disimpulkan bahwa:

1. Hasil clustering dengan **K = {int(best_k)}** memperoleh **Silhouette Score rata-rata sebesar {silhouette_avg:.4f}**,
   yang termasuk dalam kategori **{ket_sil}** menurut tabel interpretasi Silhouette Score.
2. Dari seluruh **{n_data_sil} kabupaten/kota** yang dianalisis, terdapat **{n_negatif} data ({pct_negatif:.1f}%)**
   dengan nilai Silhouette negatif, yaitu data yang berada pada area tumpang tindih antar cluster.
3. Pada perbandingan kandidat K=2, K=3, dan K=4, K final yang dipakai **{teks_status_tertinggi}**.
   Distribusi anggota menjadi syarat kelayakan sebelum Silhouette Score dibandingkan (lihat bagian 6).
4. Secara keseluruhan, hasil clustering pada penelitian ini **{teks_reliabilitas}**, dengan tetap
   mempertimbangkan konteks jumlah data (kabupaten/kota) yang relatif terbatas.
""")

    else:
        st.error("Mismatch: jumlah data silhouette tidak sama dengan jumlah data cluster")

    st.markdown(r"""
**Rumus Silhouette Score:**

$$
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
$$

Keterangan:

- $a(i)$ = rata-rata jarak data ke anggota cluster yang sama.
- $b(i)$ = rata-rata jarak data ke cluster terdekat lainnya.
- $s(i)$ = nilai silhouette untuk satu data.

Pada penelitian ini, **Elbow Method** digunakan untuk membentuk beberapa **K kandidat** berdasarkan area titik siku WCSS.
Setelah kandidat K diperoleh, **Silhouette Score** digunakan untuk mengevaluasi kualitas pemisahan cluster pada setiap kandidat,
sedangkan distribusi anggota cluster digunakan untuk melihat apakah hasil cluster masih mudah diinterpretasikan.
K final dipilih dari K=2, K=3, dan K=4. Kandidat dengan cluster tunggal tidak menjadi pilihan utama;
di antara kandidat yang distribusinya layak, dipilih Silhouette Score tertinggi. Kurva Elbow/WCSS tetap
ditampilkan untuk menunjukkan perubahan WCSS, bukan sebagai keputusan tunggal.
""")

    # ---------------------------------------------------------------
    # CONTOH PERHITUNGAN NYATA a(i), b(i), s(i) DARI SATU DATA ASLI
    # (bukan angka ilustrasi -- dihitung langsung dari X_scaled & Cluster
    # hasil K-Means, supaya bisa dipertanggungjawabkan saat sidang)
    # ---------------------------------------------------------------
    label_arr = df_result["Cluster"].values
    idx_contoh = int(np.argmin(np.abs(sample_silhouette_values - silhouette_avg)))
    nama_kab_contoh = df_result["Kabupaten"].values[idx_contoh]
    cluster_contoh = int(label_arr[idx_contoh])
    x_contoh = X_scaled[idx_contoh]

    mask_cluster_sendiri = (label_arr == cluster_contoh)
    mask_cluster_sendiri[idx_contoh] = False
    jarak_ke_cluster_sendiri = np.linalg.norm(X_scaled[mask_cluster_sendiri] - x_contoh, axis=1)
    a_i = float(jarak_ke_cluster_sendiri.mean()) if mask_cluster_sendiri.sum() > 0 else 0.0

    rata2_jarak_cluster_lain = []
    for cl_lain in sorted(set(label_arr.tolist()) - {cluster_contoh}):
        mask_lain = (label_arr == cl_lain)
        jarak_lain = np.linalg.norm(X_scaled[mask_lain] - x_contoh, axis=1)
        rata2_jarak_cluster_lain.append((cl_lain, float(jarak_lain.mean())))
    cluster_terdekat, b_i = min(rata2_jarak_cluster_lain, key=lambda t: t[1])

    penyebut = max(a_i, b_i) if max(a_i, b_i) > 0 else 1.0
    s_i_manual = (b_i - a_i) / penyebut
    s_i_sklearn = float(sample_silhouette_values[idx_contoh])

    st.markdown(f"""
### Contoh Perhitungan Silhouette Score 

Contoh diambil dari **{nama_kab_contoh}**, yang berada pada **Cluster {cluster_contoh}**.
Data ini dipilih karena nilai Silhouette-nya paling mendekati rata-rata keseluruhan
({silhouette_avg:.4f}), sehingga cukup mewakili kondisi umum data penelitian.

**Menghitung $a(i)$** — rata-rata jarak Euclidean {nama_kab_contoh} terhadap
**{int(mask_cluster_sendiri.sum())} anggota lain** pada Cluster {cluster_contoh} yang sama
(dihitung dari data terstandardisasi pada lima fitur harga):

$$
a(i) = {a_i:.4f}
$$

**Menghitung $b(i)$** — rata-rata jarak Euclidean {nama_kab_contoh} terhadap anggota Cluster
{cluster_terdekat} (cluster tetangga terdekat, dibandingkan seluruh cluster lain):

$$
b(i) = {b_i:.4f}
$$

**Perhitungan akhir:**

$$
s(i)=\\frac{{b(i)-a(i)}}{{\\max(a(i),b(i))}}=\\frac{{{b_i:.4f}-{a_i:.4f}}}{{{penyebut:.4f}}}={s_i_manual:.4f}
$$

Nilai perhitungan manual ini ({s_i_manual:.4f}) sama dengan nilai yang dihasilkan otomatis oleh
`silhouette_samples()` dari Scikit-learn untuk data yang sama, yaitu **{s_i_sklearn:.4f}**
(selisih hanya karena pembulatan). Ini membuktikan bahwa perhitungan Silhouette pada program
memang mengikuti rumus di atas, dihitung langsung dari data penelitian — bukan angka ilustrasi.
""")

    fig, ax = plt.subplots(figsize=(5, 3))
    # Gunakan label Cluster final yang sudah diurutkan berdasarkan centroid harga
    # agar nomor cluster pada silhouette sama dengan tabel hasil dan plot PCA.
    plot_silhouette(ax, X_scaled, df_result["Cluster"].values, int(best_k))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Audit Asal Angka pada Grafik Silhouette Final")
    evaluasi_final_audit = evaluasi_kandidat_lengkap(
        X_scaled=X_scaled,
        nama_objek=df_feat["Kabupaten"].reset_index(drop=True),
        k=int(best_k),
    )
    ringkasan_final_audit = evaluasi_final_audit["ringkasan"].copy()
    bukti_final_audit = evaluasi_final_audit["bukti"].copy()
    detail_final_audit = evaluasi_final_audit["detail"].copy()

    centroids_final_original = scaler.inverse_transform(evaluasi_final_audit["centroids"])
    centroid_harga_final = centroids_final_original[:, feature_cols.index("Harga_Rata2")]
    ringkasan_final_audit.insert(
        2,
        "Centroid_Harga_Rata2",
        [round(float(v), 2) for v in centroid_harga_final],
    )
    ringkasan_final_audit.insert(
        3,
        "Dasar_Penamaan_Kategori",
        [
            f"Urutan centroid Harga_Rata2 ke-{i + 1} dari rendah ke tinggi"
            for i in range(len(ringkasan_final_audit))
        ],
    )
    bukti_final_audit = bukti_final_audit.merge(
        ringkasan_final_audit[[
            "Cluster", "Centroid_Harga_Rata2", "Dasar_Penamaan_Kategori"
        ]],
        on="Cluster",
        how="left",
    )

    st.markdown(
        "Bagian ini merupakan jejak perhitungan final: setiap angka pada label grafik dapat "
        "ditelusuri ke nama kabupaten/kota dan nilai Silhouette yang membentuknya."
    )
    st.dataframe(ringkasan_final_audit, use_container_width=True, hide_index=True)
    st.dataframe(bukti_final_audit, use_container_width=True, hide_index=True)
    st.code(evaluasi_final_audit["rumus_total"], language=None)
    st.dataframe(detail_final_audit, use_container_width=True, hide_index=True)

    total_n_final_audit = int(ringkasan_final_audit["Jumlah_Anggota_n"].sum())
    selisih_final_audit = abs(
        float(evaluasi_final_audit["silhouette_avg"]) -
        float(evaluasi_final_audit["silhouette_avg_tertimbang"])
    )
    st.success(
        f"Verifikasi final: total anggota = **{total_n_final_audit}**; jumlah objek = **{len(df_feat)}**; "
        f"Silhouette rata-rata langsung = **{evaluasi_final_audit['silhouette_avg']:.6f}**; "
        f"hasil rata-rata tertimbang = **{evaluasi_final_audit['silhouette_avg_tertimbang']:.6f}**; "
        f"selisih = **{selisih_final_audit:.10f}**."
    )
    st.warning(
        "Perbedaan istilah yang harus dijelaskan saat bimbingan: kategori Rendah/Sedang/Tinggi "
        "ditentukan oleh urutan centroid Harga_Rata2, **bukan** oleh banyaknya anggota."
    )

    # =========================================================
    # 8) VISUALISASI HASIL CLUSTER (SCATTER ATRIBUT K-MEANS)
    # =========================================================
st.markdown("---")
with st.expander("9) Visualisasi Hasil Clustering (Scatter Atribut K-Means)", expanded=False):

    pca_scatter = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca_scatter.fit_transform(X_scaled)
    var_pc1 = pca_scatter.explained_variance_ratio_[0] * 100
    var_pc2 = pca_scatter.explained_variance_ratio_[1] * 100

    st.markdown(f"""
### Scatter Cluster Berdasarkan Atribut K-Means

K-Means di sini memakai **lima fitur harga sekaligus** (Harga_Rata2, Harga_Min, Harga_Max, Harga_Std,
Harga_Skewness), sehingga tidak bisa digambar langsung sebagai satu scatter 2D. Sumbu X dan Y pada
scatter berikut adalah **PC1** dan **PC2** hasil reduksi dimensi PCA (Principal Component Analysis)
dari kelima fitur tersebut, sama seperti pendekatan pada contoh cluster plot di pedoman bimbingan.

| Sumbu | Atribut | Keterangan |
|---|---|---|
| X | **PC1** | menjelaskan {var_pc1:.1f}% variasi data |
| Y | **PC2** | menjelaskan {var_pc2:.1f}% variasi data |
""")

    df_scatter = df_result.copy().reset_index(drop=True)
    df_scatter["PC1"] = pca_coords[:, 0]
    df_scatter["PC2"] = pca_coords[:, 1]

    df_scatter = df_scatter.sort_values(
        ["Cluster", "PC2", "PC1", "Kabupaten"],
        ascending=[True, False, True, True]
    ).reset_index(drop=True)
    df_scatter["No_Titik"] = np.arange(1, len(df_scatter) + 1)

    scatter_colors = ["#F8766D", "#00BA38", "#619CFF", "#C77CFF", "#00BFC4", "#FF9E4A", "#B79F00", "#F564E3", "#00A9FF"]
    scatter_markers = ["o", "^", "s", "D", "P", "X", "v", "<", ">"]
    _rentang_pc1 = df_scatter["PC1"].max() - df_scatter["PC1"].min() or 1.0
    _rentang_pc2 = df_scatter["PC2"].max() - df_scatter["PC2"].min() or 1.0
    _o = 0.03
    label_offsets = [
        (_rentang_pc1 * _o, _rentang_pc2 * _o), (-_rentang_pc1 * _o, _rentang_pc2 * _o),
        (_rentang_pc1 * _o, -_rentang_pc2 * _o), (-_rentang_pc1 * _o, -_rentang_pc2 * _o),
        (0.0, _rentang_pc2 * _o * 1.6)
    ]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FAFC")
    ax.grid(True, color="#CBD5E1", linewidth=0.75, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    clusters_sorted = sorted(df_scatter["Cluster"].dropna().astype(int).unique())
    for idx_cluster, cluster_id in enumerate(clusters_sorted):
        color = scatter_colors[idx_cluster % len(scatter_colors)]
        marker = scatter_markers[idx_cluster % len(scatter_markers)]
        sub = df_scatter[df_scatter["Cluster"].astype(int) == int(cluster_id)].copy()

        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=115,
            color=color,
            marker=marker,
            edgecolors="#0F172A",
            linewidths=0.65,
            alpha=0.95,
            label=f"Cluster {int(cluster_id)} ({kategori_map.get(int(cluster_id), '')})",
            zorder=3,
        )

        cx = float(sub["PC1"].mean()) if len(sub) else 0.0
        cy = float(sub["PC2"].mean()) if len(sub) else 0.0
        ax.scatter(cx, cy, s=235, marker="X", color="#111827", edgecolors="white", linewidths=0.9, zorder=5)
        ax.annotate(
            f"Centroid C{int(cluster_id)}",
            (cx, cy), textcoords="offset points", xytext=(7, -12),
            fontsize=8, fontweight="bold", color="#111827", zorder=6
        )

        for j, (_, row) in enumerate(sub.iterrows()):
            dx, dy = label_offsets[j % len(label_offsets)]
            ax.text(
                float(row["PC1"]) + dx,
                float(row["PC2"]) + dy,
                str(int(row["No_Titik"])),
                fontsize=8,
                fontweight="bold",
                color="#111827",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="#FFFFFF", ec="#64748B", alpha=0.78),
                zorder=7,
            )

    ax.set_title(f"Scatter Cluster Berdasarkan Atribut K-Means | K={int(best_k)} | Silhouette={float(silhouette_avg):.4f}", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% variasi)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% variasi)")
    pad_x = (df_scatter["PC1"].max() - df_scatter["PC1"].min()) * 0.12 or 0.1
    pad_y = (df_scatter["PC2"].max() - df_scatter["PC2"].min()) * 0.12 or 0.1
    ax.set_xlim(df_scatter["PC1"].min() - pad_x, df_scatter["PC1"].max() + pad_x)
    ax.set_ylim(df_scatter["PC2"].min() - pad_y, df_scatter["PC2"].max() + pad_y)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Keterangan Nomor Titik pada Scatter")
    kolom_scatter = [
        "No_Titik", "Kabupaten", "Cluster", "Kategori",
        "PC1", "PC2",
        "Harga_Rata2", "Harga_Min", "Harga_Max", "Harga_Std", "Harga_Skewness",
    ]
    kolom_scatter = [c for c in kolom_scatter if c in df_scatter.columns]
    df_scatter_tampil = df_scatter[kolom_scatter].copy()
    for c in ["PC1", "PC2"]:
        if c in df_scatter_tampil.columns:
            df_scatter_tampil[c] = df_scatter_tampil[c].round(4)
    st.dataframe(df_scatter_tampil.sort_values("No_Titik"), use_container_width=True, hide_index=True)

    st.markdown(r"""
**Fitur yang digunakan K-Means:**

$$
Harga\_Rata2 = mean(Harga), \quad Harga\_Min = min(Harga), \quad Harga\_Max = max(Harga)
$$

$$
Harga\_Std = std(Harga), \quad Harga\_Skewness = skew(Harga)
$$

K-Means menghitung jarak Euclidean menggunakan **lima fitur** di atas (setelah distandardisasi dengan StandardScaler).
Karena scatter di atas ditampilkan dalam bentuk 2D (PC1, PC2) melalui reduksi PCA, posisi titik pada
grafik adalah representasi terdekat dari ruang lima dimensi tempat K-Means sebenarnya bekerja.
""")

    # ---------------------------------------------------------------
    # PENJELASAN PC1 & PC2 (KONSEP + ANGKA NYATA DARI DATA PENELITIAN)
    # ---------------------------------------------------------------
    loadings = pca_scatter.components_  # shape (2, jumlah_fitur), urutan sama dengan feature_cols
    idx_pca_contoh = 0
    nama_kab_pca = df_result["Kabupaten"].values[idx_pca_contoh]
    x_asli_pca = X_scaled[idx_pca_contoh]
    mean_fitur = pca_scatter.mean_

    komponen_pc1 = loadings[0]
    kontribusi_pc1 = komponen_pc1 * (x_asli_pca - mean_fitur)
    pc1_manual = float(np.sum(kontribusi_pc1))
    pc1_program = float(pca_coords[idx_pca_contoh, 0])

    baris_rumus_pc1 = " + ".join(
        f"({komponen_pc1[i]:.4f} \\times ({x_asli_pca[i]:.4f} - {mean_fitur[i]:.4f}))"
        for i in range(len(feature_cols))
    )
    baris_bobot_pc1 = "\n".join(
        f"- **{feature_cols[i]}** → bobot **{komponen_pc1[i]:.4f}**"
        for i in range(len(feature_cols))
    )

    st.markdown(f"""
### Penjelasan PC1 dan PC2 

**PC1 dan PC2 bukan fitur asli.** Data penelitian sebenarnya punya **{len(feature_cols)} fitur**
(Harga_Rata2, Harga_Min, Harga_Max, Harga_Std, Harga_Skewness), jadi berada di ruang {len(feature_cols)}
dimensi yang tidak bisa digambar langsung. PCA mencari **kombinasi linear** dari kelima fitur tersebut
yang paling banyak menangkap variasi/perbedaan antar kabupaten, lalu kombinasi itu dijadikan sumbu baru:

- **PC1** = arah kombinasi fitur dengan variasi (perbedaan antar kabupaten) **terbesar**.
- **PC2** = arah kombinasi fitur dengan variasi terbesar **kedua**, yang tidak lagi tertangkap oleh PC1.

**Kenapa muncul angka persentase (PC1 = {var_pc1:.1f}%, PC2 = {var_pc2:.1f}%)?**
Angka ini disebut *explained variance ratio*, dihitung otomatis oleh `PCA()` dari Scikit-learn
berdasarkan sebaran nilai fitur pada data penelitian ini sendiri (bukan angka baku/tetap — akan
berubah kalau datanya beda). Artinya:

- **{var_pc1:.1f}%** dari seluruh variasi/perbedaan karakteristik harga antar kabupaten bisa dijelaskan
  hanya lewat sumbu PC1.
- **{var_pc2:.1f}%** sisanya (yang belum tertangkap PC1) dijelaskan oleh PC2.
- Totalnya **{(var_pc1 + var_pc2):.1f}%**, artinya scatter 2D di atas masih mewakili sebagian besar
  informasi dari data {len(feature_cols)} fitur asli, walau tidak 100%.

""")


    # =========================================================
    # 9) INTERPRETASI HASIL CLUSTER
    # =========================================================
st.markdown("---")
with st.expander("10) Interpretasi Hasil Cluster", expanded=False):

    summary_cluster = df_centroid.copy()
    jumlah_anggota_map = df_result.groupby("Cluster")["Kabupaten"].count()
    summary_cluster["Jumlah_Anggota"] = summary_cluster["Cluster"].map(jumlah_anggota_map)

    summary_cluster = summary_cluster.sort_values("Cluster").reset_index(drop=True)

    overall_mean_harga = df_result["Harga_Rata2"].mean() if "Harga_Rata2" in df_result.columns else np.nan

    st.markdown("### Dasar Penamaan Cluster Rendah, Sedang, dan Tinggi")
    st.caption(
        "Kategori tidak ditentukan dari banyaknya anggota. Cluster diurutkan berdasarkan centroid Harga_Rata2: "
        "centroid terendah menjadi kategori terendah, centroid tengah menjadi kategori sedang, dan centroid "
        "tertinggi menjadi kategori tertinggi."
    )
    dasar_kategori = summary_cluster[["Cluster", "Kategori", "Harga_Rata2", "Jumlah_Anggota"]].copy()
    dasar_kategori = dasar_kategori.rename(columns={
        "Harga_Rata2": "Centroid_Harga_Rata2",
        "Jumlah_Anggota": "Jumlah_Anggota_(bukan_dasar_kategori)",
    })
    dasar_kategori["Urutan_Centroid"] = range(1, len(dasar_kategori) + 1)
    dasar_kategori["Selisih_dari_Rata2_Keseluruhan"] = (
        dasar_kategori["Centroid_Harga_Rata2"] - overall_mean_harga
    ).round(2)
    st.dataframe(dasar_kategori, use_container_width=True, hide_index=True)
    st.info(
        "Jumlah anggota hanya menjelaskan distribusi cluster. Dasar label rendah/sedang/tinggi adalah "
        "urutan centroid harga rata-rata, bukan n atau jumlah kabupaten/kota."
    )

    st.markdown("### Interpretasi Detail Tiap Cluster")
    st.caption(
        "Interpretasi berikut dihitung dari lima fitur final K-Means, yaitu Harga_Rata2, Harga_Min, "
        "Harga_Max, Harga_Std, dan Harga_Skewness."
    )

    for _, r in summary_cluster.iterrows():
        cl = int(r["Cluster"])
        anggota = sorted(df_result[df_result["Cluster"] == cl]["Kabupaten"].tolist())

        harga_rata_cluster = float(r.get("Harga_Rata2", np.nan))
        if pd.notna(overall_mean_harga) and overall_mean_harga != 0 and pd.notna(harga_rata_cluster):
            selisih_pct = ((harga_rata_cluster - overall_mean_harga) / overall_mean_harga) * 100
            arah_harga = "di atas" if selisih_pct >= 0 else "di bawah"
            teks_harga = f"Rp {harga_rata_cluster:,.0f} ({abs(selisih_pct):.1f}% **{arah_harga}** rata-rata keseluruhan sebesar Rp {overall_mean_harga:,.0f})"
        else:
            teks_harga = "Tidak tersedia"

        st.markdown(f"""
#### Cluster {cl} ({r['Kategori']})

- **Jumlah anggota:** {int(r['Jumlah_Anggota'])} kabupaten/kota
- **Rata-rata harga anggota:** {teks_harga}
- **Rata-rata harga minimum anggota:** Rp {float(r.get('Harga_Min', 0)):,.0f}
- **Rata-rata harga maksimum anggota:** Rp {float(r.get('Harga_Max', 0)):,.0f}
- **Anggota:** {', '.join(anggota)}
""")

    st.markdown("### Detail Anggota Setiap Kategori")

    for k in df_result["Kategori"].unique():
        st.write(f"Kategori: {k}")
        st.dataframe(df_result[df_result["Kategori"] == k], use_container_width=True, hide_index=True)