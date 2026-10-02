"""Pipeline lengkap: clustering (notebook 1) -> data_clustering_inverse.csv -> klasifikasi (notebook 2) + statistik cerita.
Pakai: python train.py bank_transactions.csv"""
import sys, json, joblib
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

NUM = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"]
raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "bank_transactions.csv")
S = dict(n_raw=len(raw))

# ---------- 1. Clustering (sama dengan notebook) ----------
d = raw.dropna(); S["n_dropna"] = len(d)
d = d.drop_duplicates(); S["n_dedup"] = len(d)
S["null_rows"] = int(raw.isnull().any(axis=1).sum()); S["dup_rows"] = S["n_dropna"] - S["n_dedup"]
full = d.copy()
full["TransactionDate"] = pd.to_datetime(full["TransactionDate"]); full["PreviousTransactionDate"] = pd.to_datetime(full["PreviousTransactionDate"])
S.update(date_min=str(full.TransactionDate.min()), date_max=str(full.TransactionDate.max()),
         gap_days=float((full.TransactionDate - full.PreviousTransactionDate).dt.days.mean()),
         n_accounts=int(full.AccountID.nunique()), n_merchants=int(full.MerchantID.nunique()),
         pct_multi_device=float((full.groupby("AccountID").DeviceID.nunique() > 1).mean()),
         pct_multi_ip=float((full.groupby("AccountID")["IP Address"].nunique() > 1).mean()))
orig = d.drop(columns=[c for c in d.columns if any(k in c.lower() for k in ["id", "ip", "date"])]).reset_index(drop=True)
CAT = list(orig.select_dtypes(exclude="number").columns)
enc = {c: LabelEncoder().fit(orig[c]) for c in CAT}
pre = orig.copy()
for c in NUM:  # outlier IQR berurutan, persis notebook
    q1, q3 = orig[c].quantile([.25, .75]); i = q3 - q1
    orig = orig[(orig[c] >= q1 - 1.5 * i) & (orig[c] <= q3 + 1.5 * i)]
S["n_final"] = len(orig); S["login_removed"] = int((pre.loc[~pre.index.isin(orig.index), "LoginAttempts"] > 1).sum())
S["n_outlier"] = S["n_dedup"] - S["n_final"]
S["login_dist"] = pre.LoginAttempts.astype(int).value_counts().sort_index().to_dict()

enc_df = orig.copy()
for c in CAT: enc_df[c] = enc[c].transform(enc_df[c])
scaler = StandardScaler(); enc_df[NUM] = scaler.fit_transform(enc_df[NUM])
age_labels = ["rendah", "sedang", "tinggi"]
grp, age_bins = pd.qcut(enc_df.CustomerAge, 3, labels=age_labels, retbins=True)
enc_df["CustomerAgeGroup"] = LabelEncoder().fit_transform(grp)
km = KMeans(n_clusters=2, random_state=42).fit(enc_df)
lab = km.labels_
S["silhouette"] = float(silhouette_score(enc_df, lab)); S["cluster_sizes"] = np.bincount(lab).tolist()

inv = orig.copy().reset_index(drop=True)
age_edges = (age_bins * scaler.scale_[1] + scaler.mean_[1]).tolist()
inv["CustomerAgeGroup"] = pd.cut(inv.CustomerAge, [-np.inf, age_edges[1], age_edges[2], np.inf], labels=age_labels).astype(str)
inv["Target"] = lab
enc_df["Target"] = lab
enc_df.to_csv("data_clustering.csv", index=False); inv.to_csv("data_clustering_inverse.csv", index=False)

# ---------- 2. Statistik cerita ----------
feat = [c for c in enc_df.columns if c != "Target"]
c0, c1 = km.cluster_centers_[0], km.cluster_centers_[1]
sd = enc_df[feat].std().replace(0, np.nan)
S["effect"] = {f: float(((c1[i] - c0[i]) / sd[f]) if not np.isnan(sd[f]) else 0) for i, f in enumerate(feat)}
S["cities"] = {str(k): sorted(inv.loc[inv.Target == k, "Location"].unique().tolist()) for k in (0, 1)}
S["city_overlap"] = sorted(set(S["cities"]["0"]) & set(S["cities"]["1"]))
S["ari_location"] = float(adjusted_rand_score(lab, (enc_df.Location >= 22).astype(int)))
ncols = ["TransactionAmount", "CustomerAge", "TransactionDuration", "AccountBalance"]
S["sil_numeric_k2"] = float(silhouette_score(enc_df[ncols], KMeans(2, random_state=42, n_init=10).fit_predict(enc_df[ncols])))
S["sil_by_k"] = {k: float(silhouette_score(enc_df[ncols], KMeans(k, random_state=42, n_init=10).fit_predict(enc_df[ncols]))) for k in range(2, 8)}
S["by_cluster"] = inv.groupby("Target")[["TransactionAmount", "CustomerAge", "TransactionDuration", "AccountBalance"]].mean().round(2).to_dict("index")
S["occ_by_cluster"] = (pd.crosstab(inv.Target, inv.CustomerOccupation, normalize="index") * 100).round(1).to_dict("index")
S["channel_by_cluster"] = (pd.crosstab(inv.Target, inv.Channel, normalize="index") * 100).round(1).to_dict("index")
occ = inv.assign(ratio=inv.TransactionAmount / inv.AccountBalance).groupby("CustomerOccupation").agg(
    n=("TransactionAmount", "size"), umur=("CustomerAge", "mean"), saldo=("AccountBalance", "mean"),
    nominal=("TransactionAmount", "mean"), rasio_median=("ratio", "median"), durasi=("TransactionDuration", "mean")).round(3)
S["occupation"] = occ.reset_index().to_dict("records")
S["channel"] = inv.Channel.value_counts().to_dict(); S["type"] = inv.TransactionType.value_counts().to_dict()
S["amount_hist"] = np.histogram(inv.TransactionAmount, bins=20)[0].tolist(); S["amount_edges"] = np.histogram(inv.TransactionAmount, bins=20)[1].round(1).tolist()
S["age_edges"] = age_edges
# segmentasi alternatif (numerik ter-scale saja, k=4)
alt = KMeans(4, random_state=42, n_init=10).fit_predict(enc_df[ncols])
S["alt_k4_ari_occupation"] = float(adjusted_rand_score(alt, inv.CustomerOccupation))
S["login_hl_vs_rest"] = pre.assign(hl=pre.LoginAttempts > 1).groupby("hl")[["TransactionAmount", "TransactionDuration", "AccountBalance"]].mean().round(1).to_dict("index")

# ---------- 3. Klasifikasi (sama dengan notebook) ----------
X = pd.get_dummies(inv.drop(columns="Target"), columns=list(inv.drop(columns="Target").select_dtypes(exclude="number").columns), drop_first=True)
y = inv.Target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
dt = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
rf = RandomForestClassifier(random_state=42).fit(Xtr, ytr)
tuned = GridSearchCV(DecisionTreeClassifier(random_state=42), {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10],
                     "criterion": ["gini", "entropy"]}, cv=5, scoring="accuracy").fit(Xtr, ytr)
def ev(m, a, b): p = m.predict(a); return dict(accuracy=accuracy_score(b, p), precision=precision_score(b, p, average="macro"),
                                              recall=recall_score(b, p, average="macro"), f1=f1_score(b, p, average="macro"))
S["metrics"] = {"Decision Tree": ev(dt, Xte, yte), "Random Forest": ev(rf, Xte, yte), "Decision Tree (Tuned)": ev(tuned, Xte, yte)}
S["tuned_params"] = tuned.best_params_
S["split"] = dict(total=len(X), train=len(Xtr), test=len(Xte))
loc = [c for c in X.columns if c.startswith("Location_")]
S["n_location_cols"] = len(loc)
nl = [c for c in X.columns if c not in loc]
S["ablation"] = {
    "Semua fitur (notebook)": S["metrics"]["Random Forest"]["accuracy"],
    "Hanya fitur Location": accuracy_score(yte, DecisionTreeClassifier(random_state=42).fit(Xtr[loc], ytr).predict(Xte[loc])),
    "Tanpa fitur Location (Random Forest)": accuracy_score(yte, RandomForestClassifier(random_state=42).fit(Xtr[nl], ytr).predict(Xte[nl])),
    "Tanpa fitur Location (Decision Tree)": accuracy_score(yte, DecisionTreeClassifier(random_state=42).fit(Xtr[nl], ytr).predict(Xte[nl])),
}
imp = pd.Series(rf.feature_importances_, index=X.columns)
grp_imp = imp.groupby(lambda c: c.split("_")[0] if c.startswith(("Location", "Channel", "CustomerOccupation", "TransactionType", "CustomerAgeGroup")) else c).sum()
S["importance"] = grp_imp.sort_values(ascending=False).round(4).to_dict()
S["depth1_dt_acc"] = float(accuracy_score(y, DecisionTreeClassifier(max_depth=1, random_state=42).fit(inv[["Location"]].apply(lambda s: LabelEncoder().fit(sorted(orig.Location.unique())).transform(s)).values, y).predict(inv[["Location"]].apply(lambda s: LabelEncoder().fit(sorted(orig.Location.unique())).transform(s)).values)))

joblib.dump({"models": {"Decision Tree": dt, "Random Forest": rf, "Decision Tree (Tuned)": tuned.best_estimator_},
             "columns": list(X.columns), "age_edges": age_edges, "cities": sorted(inv.Location.unique()),
             "defaults": {c: float(inv[c].median()) for c in NUM}}, "models.joblib", compress=3)
json.dump(S, open("story.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: S[k] for k in ["n_final","silhouette","cluster_sizes","ari_location","sil_numeric_k2","sil_by_k","alt_k4_ari_occupation","ablation","depth1_dt_acc","login_removed","n_outlier","importance","metrics","tuned_params","cities","city_overlap","effect","by_cluster","occupation","login_hl_vs_rest"]}, indent=0, default=float)[:6500])
