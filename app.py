from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

app = FastAPI()

# Variabel global untuk menyimpan data
df_raw = None
df_clustered = None

@app.get("/")
def read_root():
    return {"message": "API Clustering UMKM Moccaso Gateau"}
@app.get("/main", response_class=HTMLResponse)
def show_tableau_dashboard():
    tableau_url = "https://public.tableau.com/views/AnalisisPenjualanProdukUMKMMoccasoGateau-AchmadRizqyPranataKelompok1IndependentProjectDataMining/Dashboard1?:embed=y&:display_count=yes&:showVizHome=no"

    html_content = f"""
    <html>
        <head>
            <title>Dashboard Tableau - Penjualan Produk UMKM Moccaso Gateau (Achmad Rizqy Pranata)</title>
        </head>
        <body style="text-align: center; margin-top: 30px;">
            <h2>Visualisasi Tableau: Penjualan Produk UMKM Moccaso Gateau (Achmad Rizqy Pranata)</h2>
            <iframe src="{tableau_url}" width="1000" height="800" frameborder="0" allowfullscreen></iframe>
        </body>
    </html>
    """
    return html_content

@app.get("/load-data/")
def load_penjualan_bersih():
    global df_raw
    try:
        df_raw = pd.read_csv("df_penjualan_valid_no_outliers.csv")
        return {
            "message": "Data berhasil dimuat.",
            "jumlah_baris": len(df_raw),
            "kolom": list(df_raw.columns)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/cluster/")
def cluster_data(n_clusters: int = Query(4, gt=1, lt=10)):
    global df_raw, df_clustered

    if df_raw is None:
        return JSONResponse(status_code=400, content={"error": "Silakan load data terlebih dahulu."})

    # Preprocessing
    X = df_raw.drop(['Rincian', 'Tanggal'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    agglo = AgglomerativeClustering(n_clusters=n_clusters)

    kmeans_labels = kmeans.fit_predict(X_df)
    agglo_labels = agglo.fit_predict(X_df)

    # Gabungkan data asli dengan hasil klaster
    df_clustered = df_raw.copy()
    df_clustered['kmeans_cluster'] = kmeans_labels
    df_clustered['agglo_cluster'] = agglo_labels

    # Simpan hasil clustering
    df_clustered.to_csv("combined_data_assoc.csv", index=False)

    # Interpretasi jika jumlah klaster = 4
    cluster_labels = {
        0: {
            "label": "Pelanggan Stabil",
            "recency": "260.95 → tidak terlalu lama, tapi juga tidak baru",
            "frequency": "44.38 → cukup sering melakukan transaksi",
            "monetary": "3.544.207 → nilai belanja cukup besar",
            "kesimpulan": "Pelanggan ini cukup aktif, namun jarak antar transaksi mulai menjauh.",
            "strategi": "Kirimkan penawaran menarik atau reminder untuk mendorong transaksi lebih rutin."
        },
        1: {
            "label": "Pelanggan Aktif & Loyal",
            "recency": "184.44 → cukup baru melakukan transaksi",
            "frequency": "47.94 → transaksi sering",
            "monetary": "3.833.643 → belanja tinggi",
            "kesimpulan": "Ini adalah pelanggan inti yang loyal dan aktif.",
            "strategi": "Berikan loyalty reward, akses eksklusif, atau personalisasi penawaran."
        },
        2: {
            "label": "Pelanggan Lama Tapi Sering & Bernilai Tinggi",
            "recency": "473.12 → sudah lama tidak bertransaksi",
            "frequency": "60.52 → sangat sering melakukan transaksi",
            "monetary": "4.839.480 → sangat besar",
            "kesimpulan": "Pelanggan ini dulunya sangat aktif dan bernilai tinggi, tapi sudah lama tidak kembali.",
            "strategi": "Jalankan win-back campaign seperti email khusus atau diskon personal agar mereka kembali."
        },
        3: {
            "label": "Pelanggan Premium Terbaru",
            "recency": "174.25 → baru-baru ini transaksi",
            "frequency": "47.90 → transaksi sering",
            "monetary": "3.934.254 → cukup besar",
            "kesimpulan": "Pelanggan yang baru tapi langsung aktif dan bernilai tinggi.",
            "strategi": "Jaga pengalaman mereka tetap positif, beri sambutan atau promosi khusus agar tetap loyal."
        }
    }

    interpretasi = {}
    if n_clusters == 4:
        for i in range(4):
            interpretasi[f"Klaster {i}"] = cluster_labels[i]
    else:
        interpretasi["note"] = "Interpretasi klaster hanya tersedia untuk 4 klaster (n_clusters=4)."

    return {
        "message": f"Clustering selesai untuk {n_clusters} klaster.",
        "file_hasil": "combined_data_assoc.csv",
        "contoh_data": df_clustered.head(5).to_dict(orient="records"),
        "interpretasi_klaster": interpretasi
    }

@app.get("/evaluasi/")
def evaluasi_clustering():
    global df_clustered

    if df_clustered is None:
        return JSONResponse(status_code=400, content={"error": "Belum dilakukan clustering."})

    X_only = df_clustered.drop(columns=["Rincian", "Tanggal", "kmeans_cluster", "agglo_cluster"])

    # Evaluasi metrik
    sil_k = silhouette_score(X_only, df_clustered['kmeans_cluster'])
    ch_k = calinski_harabasz_score(X_only, df_clustered['kmeans_cluster'])
    db_k = davies_bouldin_score(X_only, df_clustered['kmeans_cluster'])

    sil_h = silhouette_score(X_only, df_clustered['agglo_cluster'])
    ch_h = calinski_harabasz_score(X_only, df_clustered['agglo_cluster'])
    db_h = davies_bouldin_score(X_only, df_clustered['agglo_cluster'])

    return {
        "KMeans": {
            "Silhouette Score": round(sil_k, 5),
            "Calinski-Harabasz Score": round(ch_k, 2),
            "Davies-Bouldin Score": round(db_k, 5)
        },
        "Agglomerative Clustering": {
            "Silhouette Score": round(sil_h, 5),
            "Calinski-Harabasz Score": round(ch_h, 2),
            "Davies-Bouldin Score": round(db_h, 5)
        }
    }