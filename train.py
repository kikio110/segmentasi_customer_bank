"""Pipeline: pembersihan -> clustering (segmentasi) -> klasifikasi untuk inferensi. Pakai: python train.py bank_transactions.csv"""
import sys, json, joblib, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from features import USE, seg_features, build_X
warnings.filterwarnings("ignore")

raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "bank_transactions.csv")
M = dict(n_raw=len(raw))
# ---- 1. Pembersihan: ID/tanggal tidak dipakai, jadi hanya baris yang kosong di fitur analisis yang dibuang ----
d = raw.drop_duplicates(); M["n_dedup"] = len(d)
d = d.dropna(subset=USE).reset_index(drop=True); M["n_clean"] = len(d)
d["Ratio"] = d.TransactionAmount / d.AccountBalance
d["LoginBerulang"] = d.LoginAttempts > 1                      # dipertahankan sebagai sinyal, bukan dibuang sebagai outlier
bounds = {c: (float(d[c].quantile(.01)), float(d[c].quantile(.99))) for c in ["AccountBalance", "Ratio"]}

# ---- 2. Clustering ----
Z0 = seg_features(d.CustomerAge, d.AccountBalance, d.TransactionAmount, bounds)
scaler = StandardScaler().fit(Z0); Z = scaler.transform(Z0)
rng = np.random.RandomState(0); M["k_sweep"], M["stability"] = {}, {}
for k in range(2, 8):
    M["k_sweep"][k] = float(silhouette_score(Z, KMeans(k, random_state=42, n_init=20).fit_predict(Z)))
for k in (3, 4, 5):
    base = KMeans(k, random_state=42, n_init=20).fit(Z)
    M["stability"][k] = float(np.mean([adjusted_rand_score(base.labels_, KMeans(k, random_state=i, n_init=10).fit(Z[(ix := rng.choice(len(Z), len(Z)))]).predict(Z)) for i in range(20)]))
K = 4
km = KMeans(K, random_state=42, n_init=20).fit(Z); d["raw_lab"] = km.labels_
M["silhouette"] = float(silhouette_score(Z, km.labels_))
p = d.groupby("raw_lab").agg(age=("CustomerAge", "mean"), bal=("AccountBalance", "median"), ratio=("Ratio", "median"))
agg, sen = p.ratio.idxmax(), p.age.idxmax()
rest = [i for i in p.index if i not in (agg, sen)]
pro = max(rest, key=lambda i: p.bal[i]); muda = [i for i in rest if i != pro][0]
order = [pro, sen, muda, agg]; NAMES = ["Profesional Produktif", "Senior Mapan", "Muda Berkembang", "Pengeluaran Agresif"]
idmap = {int(raw_i): new for new, raw_i in enumerate(order)}
d["SegmentID"] = d.raw_lab.map(idmap); d["Segment"] = d.SegmentID.map(dict(enumerate(NAMES)))
d.drop(columns="raw_lab").to_csv("data_segmentasi.csv", index=False)
M["names"] = NAMES

# ---- 3. Klasifikasi untuk inferensi segmen ----
X, y = build_X(d), d.SegmentID
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
dt = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
rf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
tuned = GridSearchCV(DecisionTreeClassifier(random_state=42), {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10],
                     "criterion": ["gini", "entropy"]}, cv=5, scoring="accuracy").fit(Xtr, ytr)
models = {"Random Forest": rf, "Decision Tree": dt, "Decision Tree (Tuned)": tuned.best_estimator_}
M["metrics"] = {n: dict(accuracy=accuracy_score(yte, m.predict(Xte)), f1=f1_score(yte, m.predict(Xte), average="macro")) for n, m in models.items()}
M["cv_rf"] = float(cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5).mean())
M["tuned_params"] = tuned.best_params_; M["split"] = dict(train=len(Xtr), test=len(Xte))
M["confusion"] = confusion_matrix(yte, rf.predict(Xte)).tolist()
imp = pd.Series(rf.feature_importances_, index=X.columns)
M["importance"] = imp.groupby(lambda c: c.split("_")[0]).sum().sort_values(ascending=False).round(4).to_dict()
joblib.dump(dict(models=models, columns=list(X.columns), scaler=scaler, km=km, bounds=bounds, idmap=idmap, names=NAMES), "models.joblib", compress=3)
json.dump(M, open("meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print({k: M[k] for k in ["n_raw", "n_dedup", "n_clean", "silhouette", "k_sweep", "stability", "metrics", "cv_rf", "importance"]})
print(d.Segment.value_counts().to_dict())
