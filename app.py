import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from features import build_X, seg_features, OCC, CH, TT

B = Path(__file__).parent
st.set_page_config(page_title="Segmentasi Nasabah Bank", page_icon="🏦", layout="wide")
NAMES = ["Profesional Produktif", "Senior Mapan", "Muda Berkembang", "Pengeluaran Agresif"]
COL = dict(zip(NAMES, ["#2a9d8f", "#264653", "#e9c46a", "#e76f51"]))
INFO = {
    "Profesional Produktif": ("Usia produktif dengan saldo terbesar. Dokter dan insinyur mendominasi.",
                              "Tawarkan investasi, deposito, dan kartu premium. Dorong layanan digital karena mereka paling siap menabung lebih banyak."),
    "Senior Mapan": ("Nasabah paling senior, saldo besar, pengeluaran kecil dibanding saldo. Didominasi pensiunan dan dokter senior.",
                     "Fokus pada keamanan dan perencanaan: tabungan berjangka, asuransi, dan layanan cabang yang nyaman."),
    "Muda Berkembang": ("Mayoritas mahasiswa dengan saldo kecil tetapi transaksi masih terkendali.",
                        "Rekrut sejak dini: rekening tabungan pemula, edukasi literasi keuangan, dan fitur target menabung."),
    "Pengeluaran Agresif": ("Kelompok kecil dengan nominal transaksi besar dibanding saldo, sebagian besar mahasiswa.",
                            "Pantau dengan notifikasi limit, tawarkan produk cicilan atau dana darurat yang terukur, dan layani sebelum muncul masalah likuiditas."),
}


@st.cache_data
def load():
    return pd.read_csv(B / "data_segmentasi.csv"), json.load(open(B / "meta.json", encoding="utf-8"))


@st.cache_resource
def models():
    return joblib.load(B / "models.joblib")


df, META = load(); MD = models()
df["Segment"] = pd.Categorical(df.Segment, NAMES)
n = len(df); share = df.Segment.value_counts(normalize=True).reindex(NAMES) * 100
bal_share = df.groupby("Segment", observed=True).AccountBalance.sum().reindex(NAMES) / df.AccountBalance.sum() * 100
vol_share = df.groupby("Segment", observed=True).TransactionAmount.sum().reindex(NAMES) / df.TransactionAmount.sum() * 100
prof = df.groupby("Segment", observed=True).agg(n=("Segment", "size"), usia=("CustomerAge", "mean"), saldo=("AccountBalance", "median"),
                                                nominal=("TransactionAmount", "mean"), rasio=("Ratio", "median"), durasi=("TransactionDuration", "mean")).reindex(NAMES)
pie = lambda t: dict(title=t)

st.title("🏦 Segmentasi Nasabah Bank: Siapa Mereka dan Bagaimana Melayaninya")
st.caption(f"{n:,} profil nasabah dikelompokkan menjadi 4 segmen berdasarkan **usia, saldo, dan intensitas pengeluaran** (nominal ÷ saldo).")
tab1, tab2, tab3 = st.tabs(["📊 Insight Segmen", "🔮 Prediksi Segmen", "🤖 Model & Metodologi"])

# =============== INSIGHT ===============
with tab1:
    mapan = ["Profesional Produktif", "Senior Mapan"]; muda = ["Muda Berkembang", "Pengeluaran Agresif"]
    k = st.columns(4)
    k[0].metric("Profil nasabah", f"{n:,}")
    k[1].metric("Segmen", "4")
    k[2].metric("Saldo di 2 segmen mapan", f"{bal_share[mapan].sum():.0f}%")
    k[3].metric("Volume transaksi segmen muda", f"{vol_share[muda].sum():.0f}%")
    st.markdown(f"> **Inti ceritanya:** nasabah bank ini terbelah oleh **fase hidup**. Dua segmen mapan ({share[mapan].sum():.0f}% nasabah) menyimpan {bal_share[mapan].sum():.0f}% saldo, "
                f"sementara dua segmen muda ({share[muda].sum():.0f}% nasabah) hanya menyimpan {bal_share[muda].sum():.0f}% saldo, tetapi menyumbang {vol_share[muda].sum():.0f}% volume transaksi.")

    st.header("1️⃣ Peta pasar: empat kelompok nasabah menurut usia dan saldo")
    fig = px.scatter(df, x="CustomerAge", y="AccountBalance", color="Segment", log_y=True, opacity=.65, category_orders={"Segment": NAMES},
                     color_discrete_map=COL, labels={"CustomerAge": "Usia", "AccountBalance": "Saldo (skala log)"}, height=480)
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"Semakin tua nasabah, semakin besar saldonya. Namun ada dua kelompok muda yang berbeda: **Muda Berkembang** (rata-rata {prof.usia['Muda Berkembang']:.0f} tahun, saldo median {prof.saldo['Muda Berkembang']:,.0f}) "
                f"dan **Pengeluaran Agresif** ({prof.usia['Pengeluaran Agresif']:.0f} tahun, saldo median hanya {prof.saldo['Pengeluaran Agresif']:,.0f}).")

    st.header("2️⃣ Kenali keempat segmen")
    cols = st.columns(4)
    for c, nm in zip(cols, NAMES):
        top = df[df.Segment == nm].CustomerOccupation.value_counts(normalize=True)
        with c.container(border=True):
            st.markdown(f"### {nm}")
            st.markdown(f"**{prof.n[nm]:,}** nasabah ({share[nm]:.0f}%)")
            st.markdown(f"- Usia rata-rata: **{prof.usia[nm]:.0f}** th\n- Saldo median: **{prof.saldo[nm]:,.0f}**\n- Nominal rata-rata: **{prof.nominal[nm]:,.0f}**\n- Profesi dominan: **{top.index[0]}** ({top.iloc[0]:.0%})")
            st.caption(INFO[nm][0])
    z = prof[["usia", "saldo", "nominal", "rasio", "durasi"]]; z = (z - z.mean()) / z.std()
    z.columns = ["Usia", "Saldo", "Nominal transaksi", "Nominal ÷ saldo", "Durasi transaksi"]
    st.plotly_chart(px.imshow(z, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, text_auto=".1f", aspect="auto",
                              title="Profil relatif tiap segmen (merah = di atas rata-rata segmen, biru = di bawah)"), width="stretch")

    st.header("3️⃣ Jumlah nasabah tidak sama dengan nilai nasabah")
    v = pd.DataFrame({"Segmen": NAMES * 3, "Persen": list(share) + list(bal_share) + list(vol_share),
                      "Ukuran": ["% nasabah"] * 4 + ["% total saldo"] * 4 + ["% volume transaksi"] * 4})
    st.plotly_chart(px.bar(v, x="Segmen", y="Persen", color="Ukuran", barmode="group", text=v.Persen.map("{:.0f}%".format)), width="stretch")
    st.markdown(f"**Senior Mapan** menyumbang {share['Senior Mapan']:.0f}% nasabah, {bal_share['Senior Mapan']:.0f}% saldo, dan {vol_share['Senior Mapan']:.0f}% volume. "
                f"**Profesional Produktif** hanya {share['Profesional Produktif']:.0f}% nasabah tetapi memegang {bal_share['Profesional Produktif']:.0f}% saldo. "
                f"Sebaliknya, **Pengeluaran Agresif** hanyalah {share['Pengeluaran Agresif']:.0f}% nasabah dengan saldo {bal_share['Pengeluaran Agresif']:.1f}%, namun menghasilkan **{vol_share['Pengeluaran Agresif']:.0f}%** dari seluruh volume transaksi.")
    st.success("**Insight:** untuk simpanan dana, Senior Mapan dan Profesional Produktif adalah kunci. Untuk aktivitas transaksi, ada kelompok kecil yang sangat aktif.")

    st.header("4️⃣ Segmen Pengeluaran Agresif: kecil, tetapi paling aktif")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(prof.reset_index(), x="Segment", y="nominal", color="Segment", color_discrete_map=COL, title="Rata-rata nominal transaksi").update_layout(showlegend=False, xaxis_title=""), width="stretch")
    b.plotly_chart(px.bar(prof.reset_index(), x="Segment", y="rasio", color="Segment", color_discrete_map=COL, title="Nominal ÷ saldo (median)").update_layout(showlegend=False, xaxis_title=""), width="stretch")
    ag = prof.loc["Pengeluaran Agresif"]; ov = df.TransactionAmount.mean()
    st.markdown(f"Rata-rata nominal mereka **{ag.nominal:,.0f}**, sekitar **{ag.nominal / ov:.1f}×** rata-rata seluruh nasabah ({ov:,.0f}). Nilai transaksi mediannya **{ag.rasio:.1f}×** saldo akun, "
                f"sedangkan Senior Mapan hanya {prof.rasio['Senior Mapan']:.2f}×. Segmen ini adalah peluang pendapatan (volume tinggi) sekaligus kelompok yang perlu dipantau likuiditasnya.")

    st.header("5️⃣ Segmen ditentukan fase hidup, bukan sekadar profesi")
    ct = pd.crosstab(df.CustomerOccupation, df.Segment, normalize="index").reindex(columns=NAMES) * 100
    cl = ct.reset_index().melt(id_vars="CustomerOccupation", var_name="Segmen", value_name="Persen")
    st.plotly_chart(px.bar(cl, x="CustomerOccupation", y="Persen", color="Segmen", color_discrete_map=COL, category_orders={"Segmen": NAMES},
                           labels={"CustomerOccupation": "Profesi"}, title="Sebaran segmen di dalam tiap profesi (%)"), width="stretch")
    st.markdown(f"Satu profesi bisa tersebar di beberapa segmen. **Dokter**: {ct.loc['Doctor', 'Senior Mapan']:.0f}% ada di Senior Mapan dan {ct.loc['Doctor', 'Profesional Produktif']:.0f}% di Profesional Produktif, "
                f"yang dibedakan oleh usia. **Mahasiswa**: {ct.loc['Student', 'Muda Berkembang']:.0f}% di Muda Berkembang dan {ct.loc['Student', 'Pengeluaran Agresif']:.0f}% di Pengeluaran Agresif, "
                f"yang dibedakan oleh seberapa besar transaksinya terhadap saldo.")

    st.header("6️⃣ Pilihan kanal dan jenis transaksi relatif seragam")
    chn = pd.crosstab(df.Segment, df.Channel, normalize="index").reindex(NAMES) * 100
    login = df.groupby("Segment", observed=True).LoginBerulang.mean().reindex(NAMES) * 100
    a, b = st.columns(2)
    cm = chn.reset_index().melt(id_vars="Segment", var_name="Kanal", value_name="Persen")
    a.plotly_chart(px.bar(cm, x="Segment", y="Persen", color="Kanal", barmode="stack", title="Pangsa kanal per segmen (%)").update_layout(xaxis_title=""), width="stretch")
    b.plotly_chart(px.bar(x=NAMES, y=login.values, color=NAMES, color_discrete_map=COL, title="Transaksi dengan login berulang (>1x) per segmen (%)", labels={"x": "", "y": "%"}).update_layout(showlegend=False), width="stretch")
    spread = (chn.max() - chn.min()).max()
    st.markdown(f"Ketiga kanal (ATM, Cabang, Online) dipakai hampir merata di semua segmen; selisih terbesar antar-segmen hanya **{spread:.0f} poin persentase**. "
                f"Transaksi dengan login berulang juga tersebar di kisaran **{login.min():.1f}–{login.max():.1f}%** pada tiap segmen. "
                f"Artinya, **pembeda segmen adalah profil keuangan, bukan kanal**, sehingga strategi sebaiknya berbasis produk dan layanan, bukan hanya mendorong kanal tertentu.")

    st.header("7️⃣ Strategi yang disarankan per segmen")
    st.caption("Rekomendasi ini adalah interpretasi bisnis atas profil data di atas, bukan hasil model.")
    st.dataframe(pd.DataFrame({"Segmen": NAMES, "Ukuran": [f"{share[s]:.0f}%" for s in NAMES], "Saran strategi": [INFO[s][1] for s in NAMES]}), hide_index=True, width="stretch")

# =============== PREDIKSI ===============
with tab2:
    st.subheader("Masukkan profil nasabah, dapatkan segmennya")
    with st.form("f"):
        c = st.columns(3)
        age = c[0].number_input("Usia", 18, 90, 35); bal = c[0].number_input("Saldo akun", 50.0, 50000.0, 5000.0)
        amt = c[1].number_input("Nominal transaksi", 1.0, 5000.0, 250.0); dur = c[1].number_input("Durasi transaksi (detik)", 1.0, 600.0, 120.0)
        occ = c[2].selectbox("Profesi", OCC); ch = c[2].selectbox("Kanal", CH)
        tt = c[0].selectbox("Jenis transaksi", TT); lg = c[1].number_input("Percobaan login", 1, 5, 1)
        mname = c[2].selectbox("Model", list(MD["models"]))
        ok = st.form_submit_button("Prediksi segmen", type="primary")
    if ok:
        row = pd.DataFrame([dict(CustomerAge=age, AccountBalance=bal, TransactionAmount=amt, TransactionDuration=dur, CustomerOccupation=occ,
                                 Channel=ch, TransactionType=tt, LoginAttempts=lg)])
        mdl = MD["models"][mname]; X = build_X(row)[MD["columns"]]
        pr = mdl.predict_proba(X)[0]; sid = int(mdl.classes_[pr.argmax()]); seg = NAMES[sid]
        direct = MD["idmap"][int(MD["km"].predict(MD["scaler"].transform(seg_features([age], [bal], [amt], MD["bounds"])))[0])]
        st.markdown(f"### 🎯 Segmen: **{seg}** · keyakinan {pr.max():.0%}")
        st.info(INFO[seg][0]); st.success("**Saran:** " + INFO[seg][1])
        if direct != sid: st.caption(f"Catatan: penugasan langsung K-Means menempatkan profil ini di *{NAMES[direct]}* (profil berada dekat batas dua segmen).")
        else: st.caption("Hasil ini sama dengan penugasan langsung K-Means (centroid terdekat).")
        fig = px.scatter(df, x="CustomerAge", y="AccountBalance", color="Segment", log_y=True, opacity=.35, color_discrete_map=COL, category_orders={"Segment": NAMES},
                         labels={"CustomerAge": "Usia", "AccountBalance": "Saldo (log)"}, title="Posisi nasabah ini di peta pasar")
        fig.add_trace(go.Scatter(x=[age], y=[bal], mode="markers", marker=dict(symbol="star", size=20, color="red", line=dict(width=1, color="black")), name="Nasabah ini"))
        st.plotly_chart(fig, width="stretch")

# =============== MODEL ===============
with tab3:
    st.subheader("Bagaimana segmen dibuat")
    st.markdown(f"""
1. **Pembersihan:** {META['n_raw']:,} baris → hapus duplikat → {META['n_dedup']:,} → hapus baris yang kosong pada fitur analisis → **{META['n_clean']:,} profil**. ID, IP, dan tanggal tidak dipakai. Outlier **tidak dibuang** (nilai ekstrem di-winsorize 1–99% dan di-log), dan `LoginAttempts` > 1 tetap dipertahankan.
2. **Fitur clustering:** usia, log(saldo), dan log(nominal ÷ saldo), semuanya di-scale. Kota tidak dipakai (43 nilai tanpa urutan makna).
3. **K-Means, k = 4:** silhouette **{META['silhouette']:.2f}**, kestabilan terhadap resampling (ARI) **{META['stability']['4']:.2f}**. Segmen diberi nama otomatis dari profilnya.
4. **Klasifikasi** (Random Forest, Decision Tree, Decision Tree tuning) dilatih pada fitur nasabah untuk menebak segmen nasabah baru.
""")
    ks = pd.DataFrame({"k": list(map(int, META["k_sweep"])), "Silhouette": list(META["k_sweep"].values())})
    st.plotly_chart(px.line(ks, x="k", y="Silhouette", markers=True, title="Silhouette untuk berbagai jumlah segmen").add_vline(x=4, line_dash="dot", annotation_text="dipilih"), width="stretch")
    st.caption(f"k = 4 dipilih karena paling bermakna secara bisnis dan stabil (ARI {META['stability']['4']:.2f} vs {META['stability']['3']:.2f} untuk k = 3), walaupun silhouette k kecil lebih tinggi.")
    st.subheader("Evaluasi klasifikasi (data uji)")
    m = pd.DataFrame(META["metrics"]).T.rename(columns={"accuracy": "Akurasi", "f1": "F1 (macro)"})
    st.dataframe(m.style.format("{:.1%}"), width="stretch")
    st.caption(f"Latih {META['split']['train']:,} / uji {META['split']['test']:,} (stratified). Cross-validation 5-fold Random Forest: {META['cv_rf']:.1%}. Parameter tuning: {META['tuned_params']}")
    a, b = st.columns(2)
    a.plotly_chart(px.imshow(META["confusion"], x=NAMES, y=NAMES, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Prediksi", y="Aktual"), title="Confusion matrix (Random Forest)"), width="stretch")
    im = pd.DataFrame({"Fitur": list(META["importance"]), "Importance": list(META["importance"].values())}).sort_values("Importance")
    b.plotly_chart(px.bar(im, x="Importance", y="Fitur", orientation="h", title="Fitur paling menentukan segmen"), width="stretch")
    st.caption("Segmen dibentuk dari usia, saldo, dan rasio nominal, sehingga classifier wajar menebaknya dengan akurat. Fungsinya sebagai penebak cepat untuk nasabah baru, bukan bukti bahwa segmen itu 'benar'.")
