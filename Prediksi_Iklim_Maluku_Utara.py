# app.py — DSS Iklim Maluku Utara Kabupaten Halmahera (Final)
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------
# Config
# ----------------------
st.set_page_config(page_title="DSS Iklim Maluku Utara Kabupaten Halmahera", layout="wide")
st.title("🌦️ Decision Support System Iklim - Maluku Utara Kabupaten Halmahera")
st.markdown("Mendukung literasi iklim dan berpikir komputasi calon guru fisika melalui analisis data cuaca harian.")

DATA_PATH = "Data iklim maluku utara.xlsx"  # pastikan file ini ada di folder proyek

# ----------------------
# Helper functions (DSS)
# ----------------------
def klasifikasi_cuaca(ch, matahari):
    try:
        if pd.isna(ch) and pd.isna(matahari):
            return "N/A"
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
        if pd.isna(ch) and pd.isna(matahari):
            return "N/A"
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
        if pd.isna(ch):
            return "N/A"
        return "Ya" if ch > 50 else "Tidak"
    except:
        return "N/A"

# ----------------------
# Load & preprocess data
# ----------------------
st.sidebar.header("📂 Data")
st.sidebar.info(f"Data dimuat otomatis dari file lokal: `{DATA_PATH}`")

@st.cache_data
def process_data(path):
    # baca excel
    df = pd.read_excel(path, sheet_name="Data Harian - Table")
    # hilangkan duplikat kolom
    df = df.loc[:, ~df.columns.duplicated()]

    # konsistensi nama kolom
    if "kecepatan_angin" in df.columns and "FF_X" not in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})

    # tanggal
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")

    # pastikan kolom penting ada
    expected_cols = ['Tn','Tx','Tavg','kelembaban','curah_hujan','matahari','FF_X','DDD_X']
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Derivasi DSS
    df["Prediksi Cuaca"] = df.apply(lambda r: klasifikasi_cuaca(r.get("curah_hujan", np.nan), r.get("matahari", np.nan)), axis=1)
    df["Risiko Kekeringan"] = df.apply(lambda r: risiko_kekeringan(r.get("curah_hujan", np.nan), r.get("matahari", np.nan)), axis=1)
    df["Hujan Ekstrem"] = df["curah_hujan"].apply(hujan_ekstrem)

    # fitur tanggal
    df['Tahun'] = df['Tanggal'].dt.year
    df['Bulan'] = df['Tanggal'].dt.month

    return df

# muat data
try:
    data = process_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"File tidak ditemukan: {DATA_PATH}. Pastikan file ada di folder proyek.")
    st.stop()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# ----------------------
# Sidebar: tanggal filter
# ----------------------
st.sidebar.header("📅 Filter Tanggal Harian")
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

# ----------------------
# Agregasi bulanan
# ----------------------
possible_vars = ['Tn','Tx','Tavg','kelembaban','curah_hujan','matahari','FF_X','DDD_X']
available_vars = [v for v in possible_vars if v in data.columns]

if len(available_vars) == 0:
    st.error("Tidak ditemukan variabel cuaca yang diperlukan di file. Periksa nama kolom.")
    st.stop()

agg_dict = {v: 'mean' for v in available_vars}
if 'curah_hujan' in available_vars:
    agg_dict['curah_hujan'] = 'sum'

monthly_df = data[['Tahun','Bulan'] + available_vars].groupby(['Tahun','Bulan']).agg(agg_dict).reset_index()

st.subheader('📊 Data Bulanan (ringkasan)')
st.dataframe(monthly_df.head(12))

# ----------------------
# Pelatihan model per variabel
# ----------------------
st.subheader('📈 Pelatihan Model - Random Forest per Variabel')
X = monthly_df[['Tahun','Bulan']].copy()
models = {}
metrics = {}

# cek cukup data
if len(monthly_df) < 6:
    st.warning("Data historis bulanan kurang (<6 baris). Model tidak dilatih. Tambahkan lebih banyak data untuk pelatihan.")
else:
    for var in available_vars:
        try:
            y = monthly_df[var].astype(float).fillna(method='ffill').fillna(method='bfill')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            models[var] = model
            metrics[var] = {
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'r2': r2_score(y_test, pred)
            }
        except Exception as e:
            st.write(f"⛔ Gagal melatih model untuk {var}: {e}")

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

if metrics:
    for var, m in metrics.items():
        st.write(f"**{akademis_label.get(var,var)}** → RMSE: {m['rmse']:.3f} | R²: {m['r2']:.3f}")
else:
    st.info("Model belum tersedia (kurang data atau pelatihan gagal).")

# ----------------------
# Prediksi manual 1 bulan
# ----------------------
st.subheader('🔮 Prediksi Manual (1 Bulan)')
min_year = int(monthly_df['Tahun'].min()) if not monthly_df.empty else 2025
tahun_input = st.number_input('Masukkan Tahun Prediksi', min_value=2025, max_value=2100, value=2035)
bulan_input = st.selectbox('Pilih Bulan', list(range(1,13)))
input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=['Tahun','Bulan'])

st.write('### Hasil Prediksi:')
if models:
    for var in available_vars:
        if var in models:
            try:
                pred_val = models[var].predict(input_data)[0]
                st.success(f"{akademis_label.get(var,var)} bulan {bulan_input}/{tahun_input}: **{pred_val:.2f}**")
            except Exception as e:
                st.error(f"Gagal memprediksi {var}: {e}")
        else:
            st.info(f"Model untuk {var} tidak tersedia.")
else:
    st.info("Tidak ada model terlatih. Tidak dapat melakukan prediksi.")

# ----------------------
# Prediksi 2025-2075
# ----------------------
st.subheader('📆 Prediksi Otomatis 2025–2075')
future_years = list(range(2025, 2076))
future_months = list(range(1,13))
future_data = pd.DataFrame([(y,m) for y in future_years for m in future_months], columns=['Tahun','Bulan'])

if models:
    for var in available_vars:
        if var in models:
            try:
                future_data[f'Pred_{var}'] = models[var].predict(future_data[['Tahun','Bulan']])
            except Exception as e:
                future_data[f'Pred_{var}'] = np.nan
                st.write(f"⛔ Gagal prediksi untuk {var}: {e}")
        else:
            future_data[f'Pred_{var}'] = np.nan
else:
    # isi kolom prediksi sebagai NaN jika model tidak tersedia
    for var in available_vars:
        future_data[f'Pred_{var}'] = np.nan

st.dataframe(future_data.head(12))

# ----------------------
# Gabungkan untuk plotting historis vs prediksi
# ----------------------
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
label_list = [akademis_label.get(v,v) for v in available_vars]
selected_label = st.selectbox('Pilih Variabel Cuaca', label_list)

fig = px.line(
    future_data_merged[future_data_merged['Variabel']==selected_label],
    x='Tanggal', y='Nilai', color='Sumber', title=f'Tren {selected_label} Bulanan'
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Grafik harian DSS
# ----------------------
st.markdown('---')
st.subheader('📈 Grafik Harian - DSS')
col1, col2 = st.columns(2)
with col1:
    try:
        fig_suhu = px.line(data, x='Tanggal', y=[c for c in ['Tn','Tx','Tavg'] if c in data.columns], title='Tren Suhu Harian')
        st.plotly_chart(fig_suhu, use_container_width=True)
    except Exception as e:
        st.write(f"Gagal membuat plot suhu harian: {e}")
with col2:
    try:
        if 'curah_hujan' in data.columns:
            fig_hujan = px.line(data, x='Tanggal', y='curah_hujan', title='Tren Curah Hujan Harian')
            st.plotly_chart(fig_hujan, use_container_width=True)
        else:
            st.info("Kolom curah_hujan tidak ditemukan untuk plot harian.")
    except Exception as e:
        st.write(f"Gagal membuat plot curah hujan: {e}")

# ----------------------
# Tampilkan & unduh data
# ----------------------
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

