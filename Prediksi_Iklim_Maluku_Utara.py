import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="DSS Iklim Maluku Utara Kabupaten Halmahera", layout="wide")

st.title("🌦️ Decision Support System Iklim - Maluku Utara Kabupaten Halmahera")
st.markdown("Mendukung literasi iklim dan berpikir komputasi calon guru fisika melalui analisis data cuaca harian.")

# -----------------------------
# DSS helper functions
# -----------------------------

def klasifikasi_cuaca(ch, matahari):
    try:
        if ch > 20:
            return "Hujan"
        elif ch > 5:
            return "Berawan"
        elif matahari > 4:
            return "Cerah"
        else:
            return "Berawan"
    except:
        return "N/A"

def risiko_kekeringan(ch, matahari):
    try:
        if ch < 1 and matahari > 6:
            return "Risiko Tinggi"
        elif ch < 5:
            return "Risiko Sedang"
        else:
            return "Risiko Rendah"
    except:
        return "N/A"

def hujan_ekstrem(ch):
    try:
        return "Ya" if ch > 50 else "Tidak"
    except:
        return "N/A"

# -----------------------------
# -----------------------------
# Load data otomatis tanpa upload
# -----------------------------

DATA_PATH = "Data iklim maluku utara.xlsx"
st.sidebar.success("📂 Data dimuat otomatis dari file lokal: Data iklim maluku utara.xlsx")

@st.cache_data
def process_data(_):  # parameter dummy agar cache tetap bekerja
    df = pd.read_excel(DATA_PATH, sheet_name="Data Harian - Table")
    df = df.loc[:, ~df.columns.duplicated()]

    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month

    df["Prediksi Cuaca"] = df.apply(lambda row: klasifikasi_cuaca(row.get("curah_hujan", 0), row.get("matahari", 0)), axis=1)
    df["Risiko Kekeringan"] = df.apply(lambda row: risiko_kekeringan(row.get("curah_hujan", 0), row.get("matahari", 0)), axis=1)
    df["Hujan Ekstrem"] = df["curah_hujan"].apply(hujan_ekstrem)

    return df(uploaded_file_path_or_buffer):
    # baca file
    df = pd.read_excel(uploaded_file_path_or_buffer, sheet_name="Data Harian - Table")

    # pastikan nama kolom konsisten
    if "kecepatan_angin" in df.columns and "FF_X" not in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})

    # parse tanggal
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors='coerce')

    # isi kolom yang sering dipakai jika kosong
    expected_cols = ['Tn','Tx','Tavg','kelembaban','curah_hujan','matahari','FF_X','DDD_X']
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # DSS columns
    df["Prediksi Cuaca"] = df.apply(lambda row: klasifikasi_cuaca(row.get("curah_hujan", np.nan), row.get("matahari", np.nan)), axis=1)
    df["Risiko Kekeringan"] = df.apply(lambda row: risiko_kekeringan(row.get("curah_hujan", np.nan), row.get("matahari", np.nan)), axis=1)
    df["Hujan Ekstrem"] = df["curah_hujan"].apply(hujan_ekstrem)

    # tanggal features
    df['Tahun'] = df['Tanggal'].dt.year
    df['Bulan'] = df['Tanggal'].dt.month

    return df

# Load data (uploaded or default file shipped)
if uploaded_file:
    data = process_data(uploaded_file)
else:
    # gunakan nama file di server jika ada
    try:
        data = process_data("/mnt/data/Data iklim maluku utara.xlsx")
    except Exception as e:
        st.error("Tidak ada file default. Silakan unggah file Excel dengan sheet 'Data Harian - Table'.")
        st.stop()

# -----------------------------
# Sidebar: tanggal filter
# -----------------------------
st.sidebar.header("📅 Filter Tanggal")
min_date = data["Tanggal"].min()
max_date = data["Tanggal"].max()
selected_date = st.sidebar.date_input("Pilih Tanggal", value=min_date, min_value=min_date, max_value=max_date)

# Tampilkan info harian
row = data[data["Tanggal"] == pd.to_datetime(selected_date)]
if not row.empty:
    info = row.iloc[0]
    st.subheader(f"📊 Data Iklim - {pd.to_datetime(selected_date).strftime('%d %B %Y')}")
    st.write(f"- Suhu rata-rata: **{info['Tavg']}°C**")
    st.write(f"- Suhu min (Tn): **{info['Tn']}°C**")
    st.write(f"- Suhu max (Tx): **{info['Tx']}°C**")
    st.write(f"- Kelembaban: **{info['kelembaban']}%**")
    st.write(f"- Curah hujan: **{info['curah_hujan']} mm**")
    st.write(f"- Matahari: **{info['matahari']} jam**")
    st.write(f"- Kecepatan angin (FF_X): **{info.get('FF_X', 'N/A')}**")

    st.markdown("---")
    st.subheader("🤖 Hasil Analisis Sistem")
    st.success(f"**Prediksi Cuaca:** {info['Prediksi Cuaca']}")
    st.info(f"**Risiko Kekeringan:** {info['Risiko Kekeringan']}")
    st.warning(f"**Hujan Ekstrem:** {info['Hujan Ekstrem']}")
else:
    st.error("Data tidak ditemukan untuk tanggal tersebut.")

# -----------------------------
# Agregasi bulanan untuk model
# -----------------------------
possible_vars = ['Tn','Tx','Tavg','kelembaban','curah_hujan','matahari','FF_X','DDD_X']
available_vars = [v for v in possible_vars if v in data.columns]

agg_dict = {v: 'mean' for v in available_vars}
if 'curah_hujan' in available_vars:
    agg_dict['curah_hujan'] = 'sum'

monthly_df = data[['Tahun','Bulan'] + available_vars].groupby(['Tahun','Bulan']).agg(agg_dict).reset_index()

st.subheader('📊 Data Bulanan (ringkasan)')
st.dataframe(monthly_df.head(12))

# -----------------------------
# Train models per variabel
# -----------------------------
st.subheader('📈 Pelatihan Model - Random Forest per Variabel')

X = monthly_df[['Tahun','Bulan']]
models = {}
metrics = {}

for var in available_vars:
    y = monthly_df[var].fillna(method='ffill').fillna(method='bfill')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    models[var] = model
    metrics[var] = {
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'r2': r2_score(y_test, pred)
    }

# label akademis
akademis_label = {
    'Tn': 'Suhu Minimum (°C)',
    'Tx': 'Suhu Maksimum (°C)',
    'Tavg': 'Suhu Rata-rata (°C)',
    'kelembaban': 'Kelembaban Udara (%)',
    'curah_hujan': 'Curah Hujan (mm)',
    'matahari': 'Durasi Penyinaran Matahari (jam)',
    'FF_X': 'Kecepatan Angin Maksimum (m/s)',
    'DDD_X': 'Arah Angin saat Kecepatan Maksimum (°)'
}

for var, m in metrics.items():
    st.write(f"**{akademis_label.get(var,var)}** → RMSE: {m['rmse']:.3f} | R²: {m['r2']:.3f}")

# -----------------------------
# Prediksi manual 1 bulan
# -----------------------------
st.subheader('🔮 Prediksi Manual (1 Bulan)')
min_year = int(monthly_df['Tahun'].min()) if not monthly_df.empty else 2025
tahun_input = st.number_input('Masukkan Tahun Prediksi', min_value=2025, max_value=2100, value=2035)
bulan_input = st.selectbox('Pilih Bulan', list(range(1,13)))
input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=['Tahun','Bulan'])

st.write('### Hasil Prediksi:')
for var in available_vars:
    try:
        pred_val = models[var].predict(input_data)[0]
        st.success(f"{akademis_label.get(var,var)} bulan {bulan_input}/{tahun_input}: **{pred_val:.2f}**")
    except Exception as e:
        st.error(f"Gagal memprediksi {var}: {e}")

# -----------------------------
# Prediksi 2025-2075
# -----------------------------
st.subheader('📆 Prediksi Otomatis 2025–2075')
future_years = list(range(2025, 2076))
future_months = list(range(1,13))
future_data = pd.DataFrame([(y,m) for y in future_years for m in future_months], columns=['Tahun','Bulan'])
for var in available_vars:
    future_data[f'Pred_{var}'] = models[var].predict(future_data[['Tahun','Bulan']])

st.dataframe(future_data.head(12))

# -----------------------------
# Gabungkan untuk plotting historis vs prediksi
# -----------------------------
monthly_df['Sumber'] = 'Data Historis'
future_data_plot = future_data.copy()
future_data_plot['Sumber'] = 'Prediksi'

merge_list = []
for var in available_vars:
    hist = monthly_df[['Tahun','Bulan',var,'Sumber']].rename(columns={var:'Nilai'})
    hist['Variabel'] = akademis_label.get(var,var)
    fut = future_data_plot[['Tahun','Bulan',f'Pred_{var}','Sumber']].rename(columns={f'Pred_{var}':'Nilai'})
    fut['Variabel'] = akademis_label.get(var,var)
    merge_list.append(pd.concat([hist,fut], ignore_index=True))

future_data_merged = pd.concat(merge_list, ignore_index=True)
future_data_merged['Tanggal'] = pd.to_datetime(future_data_merged['Tahun'].astype(int).astype(str) + '-' + future_data_merged['Bulan'].astype(int).astype(str) + '-01')

st.subheader('📈 Grafik Tren Variabel Cuaca (Historis vs Prediksi)')
selected_label = st.selectbox('Pilih Variabel Cuaca', [akademis_label.get(v,v) for v in available_vars])

fig = px.line(
    future_data_merged[future_data_merged['Variabel']==selected_label],
    x='Tanggal', y='Nilai', color='Sumber', title=f'Tren {selected_label} Bulanan'
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Grafik harian DSS
# -----------------------------
st.markdown('---')
st.subheader('📈 Grafik Harian - DSS')
col1, col2 = st.columns(2)
with col1:
    fig_suhu = px.line(data, x='Tanggal', y=['Tn','Tx','Tavg'], title='Tren Suhu Harian')
    st.plotly_chart(fig_suhu, use_container_width=True)
with col2:
    fig_hujan = px.line(data, x='Tanggal', y='curah_hujan', title='Tren Curah Hujan Harian')
    st.plotly_chart(fig_hujan, use_container_width=True)

# -----------------------------
# Tampilkan dan unduh data
# -----------------------------
with st.expander('📁 Lihat dan Unduh Data Lengkap'):
    st.dataframe(data)

    st.markdown('⬇️ **Unduh Data Hasil Analisis (Excel & CSV):**')
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        data.to_excel(writer, sheet_name='Hasil DSS', index=False)
        monthly_df.to_excel(writer, sheet_name='Bulanan', index=False)
        future_data.to_excel(writer, sheet_name='Prediksi_2025_2075', index=False)
        writer.close()
    st.download_button(label='Unduh Excel', data=buffer.getvalue(), file_name='hasil_dss_iklim.xlsx', mime='application/vnd.ms-excel')

    # CSV prediksi
    csv = future_data.to_csv(index=False).encode('utf-8')
    st.download_button(label='📥 Download CSV Prediksi 2025–2075', data=csv, file_name='prediksi_cuaca_multi_variabel_2025_2075.csv', mime='text/csv')

st.markdown('---')
st.write('📌 Catatan: requirements yang dipakai — streamlit, pandas, numpy, scikit-learn, plotly, openpyxl, xlsxwriter.')
