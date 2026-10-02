import json
from pathlib import Path
import joblib, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st

B = Path(__file__).parent
st.set_page_config(page_title="Cerita Data Transaksi Bank", page_icon="🏦", layout="wide")


@st.cache_data
def load_story():
    return json.load(open(B / "story.json", encoding="utf-8"))


@st.cache_resource
def load_models():
    return joblib.load(B / "models.joblib")


S, M = load_story(), load_models()
CL = {"0": "#4c78a8", "1": "#f58518"}
occ = pd.DataFrame(S["occupation"]).rename(columns={"CustomerOccupation": "Profesi"})
pct = lambda a, b: abs(a - b) / ((a + b) / 2) * 100
bc = S["by_cluster"]

st.title("🏦 Cerita di Balik Data Transaksi Bank")
st.caption("Dari data mentah → clustering → klasifikasi. Setiap bagian ditulis sebagai temuan, bukan sekadar angka.")
tab1, tab2, tab3 = st.tabs(["📖 Cerita Data", "🔮 Inferensi Cluster", "🤖 Evaluasi Model"])

# ======================= CERITA =======================
with tab1:
    k = st.columns(4)
    k[0].metric("Transaksi mentah", f"{S['n_raw']:,}")
    k[1].metric("Dipakai untuk model", f"{S['n_final']:,}", f"-{(1 - S['n_final'] / S['n_raw']):.0%}", delta_color="off")
    k[2].metric("Akun unik", f"{S['n_accounts']:,}")
    k[3].metric("Silhouette (notebook)", f"{S['silhouette']:.2f}")
    st.markdown("> **Ringkasan cerita:** ada pola nyata di data (profesi menentukan usia, saldo, dan tekanan finansial), "
                "tetapi dua cluster yang dihasilkan tidak menangkapnya. Cluster itu ternyata hanya membagi **daftar kota** secara alfabetis, "
                "dan klasifikasi 100% hanya mengulang aturan itu.")

    # ---- Bab 1
    st.header("1️⃣ Hampir seperempat data hilang sebelum model dilatih")
    a, b = st.columns([3, 2])
    steps = ["Data mentah", "Hapus nilai kosong", "Hapus duplikat", "Hapus outlier (IQR)"]
    vals = [S["n_raw"], S["n_raw"] - S["null_rows"], S["n_dedup"], S["n_final"]]
    a.plotly_chart(px.bar(x=steps, y=vals, text=vals, labels={"x": "", "y": "Jumlah baris"}, title="Jumlah baris di tiap tahap pembersihan"),
                   width="stretch")
    with b:
        st.markdown(f"Dari **{S['n_raw']:,}** baris, **{S['null_rows']}** punya nilai kosong, **{S['dup_rows']}** duplikat, dan **{S['n_outlier']}** dibuang sebagai outlier. "
                    f"Tersisa **{S['n_final']:,}** baris ({S['n_final'] / S['n_raw']:.0%}).")
        st.info(f"Perhatikan angka ini: dari {S['n_outlier']} outlier, **{S['login_removed']}** dibuang karena `LoginAttempts` > 1. Kita kembali ke ini di bab 3.")

    # ---- Bab 2
    st.header("2️⃣ Profesi nasabah menceritakan tiga kehidupan finansial yang berbeda")
    st_ = occ.set_index("Profesi")
    mult = st_.loc["Student", "rasio_median"] / st_.loc["Doctor", "rasio_median"]
    c = st.columns(3)
    c[0].plotly_chart(px.bar(occ, x="Profesi", y="umur", color="Profesi", title="Rata-rata usia", labels={"umur": "tahun"}).update_layout(showlegend=False), width="stretch")
    c[1].plotly_chart(px.bar(occ, x="Profesi", y="saldo", color="Profesi", title="Rata-rata saldo akun", labels={"saldo": "saldo"}).update_layout(showlegend=False), width="stretch")
    c[2].plotly_chart(px.bar(occ, x="Profesi", y="rasio_median", color="Profesi", title="Nominal transaksi ÷ saldo (median)", labels={"rasio_median": "rasio"}).update_layout(showlegend=False), width="stretch")
    st.markdown(f"Empat profesi tersebar merata ({', '.join(f'{r.Profesi} {int(r.n)}' for r in occ.itertuples())}), tetapi hidupnya jauh berbeda. "
                f"**Dokter** (rata-rata {st_.loc['Doctor', 'umur']:.0f} tahun) menyimpan saldo ±{st_.loc['Doctor', 'saldo']:,.0f}, sedangkan **mahasiswa** (±{st_.loc['Student', 'umur']:.0f} tahun) hanya ±{st_.loc['Student', 'saldo']:,.0f}. "
                f"Padahal nominal transaksi mahasiswa justru yang tertinggi ({st_.loc['Student', 'nominal']:.0f}). Akibatnya, sekali transaksi mahasiswa memakai porsi saldo **±{mult:.1f}× lebih besar** daripada dokter.")
    st.success("**Insight bisnis:** inilah segmentasi yang bermakna. Mahasiswa adalah kelompok paling rentan secara arus kas, dan pemantauannya sebaiknya memakai rasio nominal terhadap saldo, bukan nominal saja.")

    # ---- Bab 3
    st.header("3️⃣ Sinyal yang paling mirip penipuan justru dibuang sebagai 'outlier'")
    a, b = st.columns([2, 3])
    ld = pd.DataFrame({"Percobaan login": list(S["login_dist"]), "Transaksi": list(S["login_dist"].values())})
    ld["Dibuang?"] = ld["Percobaan login"].apply(lambda v: "Ya, semua" if int(v) > 1 else "Tidak")
    a.plotly_chart(px.bar(ld, x="Percobaan login", y="Transaksi", color="Dibuang?", text="Transaksi", title="Distribusi LoginAttempts sebelum outlier dibuang",
                          color_discrete_map={"Ya, semua": "#d62728", "Tidak": "#9aa5b1"}), width="stretch")
    h = S["login_hl_vs_rest"]
    with b:
        st.markdown(f"Sebanyak **{S['login_removed']} transaksi** ({S['login_removed'] / S['n_dedup']:.1%}) memiliki 2–5 kali percobaan login. Dalam deteksi penipuan, inilah sinyal paling klasik. "
                    f"Metode IQR menganggap semuanya outlier karena 95% transaksi hanya 1 kali login, sehingga **seluruhnya terbuang**. Setelah itu `LoginAttempts` bernilai 1 untuk semua baris dan tidak punya pengaruh apa pun pada model.")
        st.markdown(f"Nominal transaksi kelompok ini hampir sama dengan yang lain ({h['true']['TransactionAmount']:.0f} vs {h['false']['TransactionAmount']:.0f}), "
                    f"jadi mereka **tidak bisa dikenali dari nominalnya**. Hanya perilaku loginnya yang berbeda.")
        st.markdown(f"Fitur lain juga tidak banyak membantu: **{S['pct_multi_device']:.0%}** akun memakai lebih dari satu perangkat dan IP, jadi hampir semua akun tampak 'mencurigakan'. "
                    f"Seluruh `TransactionDate` jatuh dalam rentang beberapa menit (`{S['date_min'][:16]}` s.d. `{S['date_max'][11:16]}`), sehingga pola waktu tidak bisa dianalisis.")
    st.warning("**Rekomendasi:** perlakukan `LoginAttempts` > 1 sebagai fitur biner (`login_gagal`), bukan sebagai outlier yang dibuang.")

    # ---- Bab 4
    st.header("4️⃣ Dua cluster itu ternyata dua daftar kota, bukan dua tipe nasabah")
    eff = pd.DataFrame({"Fitur": list(S["effect"]), "Selisih antar-cluster (satuan std)": [abs(v) for v in S["effect"].values()]}).sort_values("Selisih antar-cluster (satuan std)")
    eff["Fitur"] = eff["Fitur"].replace({"Location": "Location ⚠️"})
    a, b = st.columns([3, 2])
    a.plotly_chart(px.bar(eff, x="Selisih antar-cluster (satuan std)", y="Fitur", orientation="h", title="Seberapa berbeda kedua cluster pada tiap fitur?"), width="stretch")
    with b:
        st.markdown(f"Selisih rata-rata antar-cluster pada `Location` adalah **{abs(S['effect']['Location']):.2f}** simpangan baku, sedangkan fitur lain paling besar hanya **{max(abs(v) for k_, v in S['effect'].items() if k_ != 'Location'):.2f}**. "
                    f"Penyebabnya: kota di-*label-encode* menjadi 0–42 tanpa di-scale, sehingga rentang angkanya jauh melampaui fitur lain yang sudah di-scale (±1) dan mendominasi jarak K-Means.")
        st.markdown(f"Buktinya: batas satu kode kota (≥ 22) mereproduksi label cluster dengan kecocokan **ARI = {S['ari_location']:.2f}** (sempurna). "
                    f"Cluster 0 berisi kota **{S['cities']['0'][0]} … {S['cities']['0'][-1]}**, sedangkan cluster 1 berisi **{S['cities']['1'][0]} … {S['cities']['1'][-1]}**. Jadi cluster ini hanyalah pembagian kota **A–L vs M–W** secara alfabetis.")
    with st.expander("Lihat daftar kota tiap cluster"):
        x, y = st.columns(2)
        x.markdown("**Cluster 0**: " + ", ".join(S["cities"]["0"]))
        y.markdown("**Cluster 1**: " + ", ".join(S["cities"]["1"]))
    cmp_ = pd.DataFrame(bc).T.rename(columns={"TransactionAmount": "Nominal", "CustomerAge": "Usia", "TransactionDuration": "Durasi (dtk)", "AccountBalance": "Saldo"})
    cmp_.index = ["Cluster 0", "Cluster 1"]
    st.dataframe(cmp_, width="stretch")
    st.markdown(f"Dalam satuan aslinya, perbedaan kedua cluster sangat kecil: usia berbeda **{abs(bc['0']['CustomerAge'] - bc['1']['CustomerAge']):.1f} tahun**, durasi **{abs(bc['0']['TransactionDuration'] - bc['1']['TransactionDuration']):.1f} detik**, "
                f"dan saldo hanya **{pct(bc['0']['AccountBalance'], bc['1']['AccountBalance']):.1f}%**. Label seperti *Stable & Careful* atau *Young & Active* tidak didukung data. Itu tafsir atas selisih yang pada dasarnya acak.")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(x=["Semua fitur (notebook)", "Hanya fitur numerik"], y=[S["silhouette"], S["sil_numeric_k2"]], text=[f"{S['silhouette']:.2f}", f"{S['sil_numeric_k2']:.2f}"],
                          labels={"x": "", "y": "Silhouette"}, title="Silhouette k=2: terlihat bagus karena kota, bukan karena perilaku"), width="stretch")
    sk = pd.DataFrame({"k": list(S["sil_by_k"]), "silhouette": list(S["sil_by_k"].values())})
    a2 = px.line(sk, x="k", y="silhouette", markers=True, title="Silhouette fitur numerik untuk k = 2…7 (datar)", range_y=[0, 0.6])
    b.plotly_chart(a2, width="stretch")
    st.info(f"Pada fitur perilaku saja, silhouette hanya **{S['sil_numeric_k2']:.2f}** dan tidak membaik untuk k mana pun. Artinya data ini **tidak punya gerombol alami** pada fitur numeriknya. Struktur yang nyata ada pada profesi (bab 2).")

    # ---- Bab 5
    st.header("5️⃣ Akurasi 100% bukan prestasi model, melainkan kebocoran label")
    ab = pd.DataFrame({"Skenario": list(S["ablation"]), "Akurasi": list(S["ablation"].values())})
    a, b = st.columns([3, 2])
    a.plotly_chart(px.bar(ab, x="Akurasi", y="Skenario", orientation="h", text=ab["Akurasi"].map("{:.0%}".format), range_x=[0, 1.1],
                          title="Akurasi data uji dengan dan tanpa fitur Location").add_vline(x=0.5, line_dash="dot", annotation_text="tebakan acak"), width="stretch")
    with b:
        st.markdown(f"Label (`Target`) lahir dari fitur yang sama dengan yang diberikan ke classifier, sehingga Decision Tree, Random Forest, dan versi tuning semuanya mendapat **100%**. "
                    f"Fitur `Location` menyumbang **{S['importance']['Location']:.1%}** dari seluruh importance Random Forest.")
        st.markdown(f"Satu pohon dengan **satu pemisahan** pada kota sudah cukup untuk 100%. Begitu fitur kota dibuang, akurasi jatuh ke **{S['ablation']['Tanpa fitur Location (Random Forest)']:.0%}** (RF) dan **{S['ablation']['Tanpa fitur Location (Decision Tree)']:.0%}** (DT), setara tebakan acak.")
    st.error("Model ini tidak mempelajari 'tipe nasabah'. Ia hanya menghafal aturan *kota A–L → cluster 0, M–W → cluster 1*. Jangan dipakai untuk keputusan bisnis.")

    # ---- Bab 6
    st.header("6️⃣ Apa yang sebaiknya dilakukan selanjutnya")
    st.markdown(f"""
1. **Scale atau one-hot semua fitur kategorikal sebelum K-Means**, atau keluarkan `Location` dari clustering (43 kota terlalu banyak untuk satu fitur ordinal).
2. **Pertahankan `LoginAttempts` > 1** sebagai fitur biner dan tandai transaksi tersebut untuk ditinjau. Itu {S['login_removed']} kasus yang tidak boleh dibuang.
3. **Tambahkan fitur turunan:** rasio nominal ÷ saldo (cerita mahasiswa di bab 2) dan selisih hari dari transaksi sebelumnya.
4. **Jangan melatih classifier pada label yang dibentuk dari fitur yang sama.** Gunakan label eksternal (misal status fraud) atau evaluasi stabilitas cluster dengan cara lain.
5. Cek apakah segmentasi berbasis profesi lebih bermakna: K-Means k=4 pada fitur numerik hanya cocok dengan profesi pada ARI ≈ **{S['alt_k4_ari_occupation']:.2f}**. Itu lemah, jadi perlu fitur yang lebih kaya.
""")

# ======================= INFERENSI =======================
def build_X(v):
    row = {c: 0 for c in M["columns"]}
    for n in ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"]: row[n] = v[n]
    e = M["age_edges"]; v["CustomerAgeGroup"] = "rendah" if v["CustomerAge"] <= e[1] else "sedang" if v["CustomerAge"] <= e[2] else "tinggi"
    for col in ["TransactionType", "Location", "Channel", "CustomerOccupation", "CustomerAgeGroup"]:
        key = f"{col}_{v[col]}"
        if key in row: row[key] = 1
    return pd.DataFrame([row])[M["columns"]]


with tab2:
    st.subheader("Prediksi cluster untuk satu transaksi")
    d = M["defaults"]
    with st.form("f"):
        c = st.columns(3)
        v = dict(
            TransactionAmount=c[0].number_input("Nominal transaksi", 0.0, 5000.0, d["TransactionAmount"]),
            AccountBalance=c[0].number_input("Saldo akun", 0.0, 50000.0, d["AccountBalance"]),
            TransactionDuration=c[0].number_input("Durasi transaksi (detik)", 1.0, 600.0, d["TransactionDuration"]),
            CustomerAge=c[1].number_input("Usia nasabah", 18, 90, int(d["CustomerAge"])),
            CustomerOccupation=c[1].selectbox("Profesi", ["Doctor", "Engineer", "Retired", "Student"]),
            LoginAttempts=c[1].number_input("Percobaan login", 1, 5, 1),
            Location=c[2].selectbox("Kota", M["cities"], index=M["cities"].index("Houston")),
            Channel=c[2].selectbox("Kanal", ["ATM", "Branch", "Online"]),
            TransactionType=c[2].selectbox("Jenis", ["Debit", "Credit"]),
        )
        mname = st.selectbox("Model", list(M["models"]), index=1)
        go_ = st.form_submit_button("Prediksi", type="primary")
    if go_:
        model = M["models"][mname]; X = build_X(dict(v))
        pred = int(model.predict(X)[0]); pr = model.predict_proba(X)[0].max()
        st.markdown(f"### Cluster **{pred}** · keyakinan {pr:.0%}")
        if v["LoginAttempts"] > 1:
            st.warning("Model dilatih hanya dengan `LoginAttempts = 1` (sisanya dibuang sebagai outlier), jadi nilai ini tidak berpengaruh pada prediksi.")
        # uji sensitivitas: apa yang menggeser prediksi?
        others = {"Nominal": ("TransactionAmount", [10, 100, 400, 900]), "Saldo": ("AccountBalance", [200, 2000, 8000, 15000]),
                  "Usia": ("CustomerAge", [20, 35, 50, 70]), "Durasi": ("TransactionDuration", [20, 80, 160, 280]),
                  "Profesi": ("CustomerOccupation", ["Doctor", "Engineer", "Retired", "Student"]), "Kanal": ("Channel", ["ATM", "Branch", "Online"])}
        rows = []
        for name, (col, vals) in others.items():
            ps = [int(model.predict(build_X({**v, col: x}))[0]) for x in vals]
            rows.append((f"Ubah {name}", f"{sum(p != pred for p in ps)} dari {len(ps)} skenario berubah"))
        ps = [int(model.predict(build_X({**v, "Location": cty}))[0]) for cty in M["cities"]]
        rows.append(("Ubah Kota", f"{sum(p != pred for p in ps)} dari {len(ps)} kota akan berubah"))
        st.markdown("**Uji sensitivitas: faktor apa yang benar-benar menentukan prediksi ini?**")
        st.table(pd.DataFrame(rows, columns=["Perubahan", "Efek pada cluster"]))
        st.info(f"Kota **{v['Location']}** berada di cluster {pred}. Prediksi hanya berubah bila kota diganti, bukan karena nominal, saldo, usia, atau profesi. Ini bukti langsung dari bab 4 dan 5.")

# ======================= EVALUASI =======================
with tab3:
    st.subheader("Evaluasi model klasifikasi (data uji)")
    st.caption(f"Data hasil clustering: {S['split']['total']:,} baris → latih {S['split']['train']:,} / uji {S['split']['test']:,} (stratified, random_state=42). Fitur one-hot, sama seperti notebook.")
    m = pd.DataFrame(S["metrics"]).T.rename(columns={"accuracy": "Akurasi", "precision": "Precision", "recall": "Recall", "f1": "F1"})
    st.dataframe(m.style.format("{:.2%}"), width="stretch")
    st.caption(f"Parameter terbaik GridSearchCV: {S['tuned_params']}")
    imp = pd.DataFrame({"Fitur": list(S["importance"]), "Importance": list(S["importance"].values())}).sort_values("Importance")
    st.plotly_chart(px.bar(imp, x="Importance", y="Fitur", orientation="h", title="Importance Random Forest (dijumlah per fitur asal)"), width="stretch")
    st.warning("Skor sempurna di atas adalah tanda **kebocoran label**, bukan kualitas model. Lihat bab 5 di tab *Cerita Data*.")
