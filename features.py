import numpy as np, pandas as pd

OCC, CH, TT = ["Doctor", "Engineer", "Retired", "Student"], ["ATM", "Branch", "Online"], ["Credit", "Debit"]
USE = ["TransactionAmount", "TransactionType", "Channel", "CustomerAge", "CustomerOccupation",
       "TransactionDuration", "LoginAttempts", "AccountBalance"]


def seg_features(age, balance, amount, bounds):
    """3 fitur clustering: usia, log saldo, log rasio (nominal/saldo) - outlier di-winsorize, bukan dibuang."""
    ratio = np.asarray(amount, float) / np.asarray(balance, float)
    lb = np.log1p(np.clip(np.asarray(balance, float), *bounds["AccountBalance"]))
    lr = np.log1p(np.clip(ratio, *bounds["Ratio"]))
    return np.column_stack([np.asarray(age, float), lb, lr])


def build_X(df):
    """Fitur klasifikasi (dipakai training & inferensi)."""
    X = pd.DataFrame({"Usia": df.CustomerAge.astype(float), "Saldo": df.AccountBalance.astype(float),
                      "Nominal": df.TransactionAmount.astype(float),
                      "Rasio": df.TransactionAmount.astype(float) / df.AccountBalance.astype(float),
                      "Durasi": df.TransactionDuration.astype(float), "LoginBerulang": (df.LoginAttempts > 1).astype(int)})
    for p, cats, col in [("Profesi", OCC, "CustomerOccupation"), ("Kanal", CH, "Channel"), ("Jenis", TT, "TransactionType")]:
        for c in cats: X[f"{p}_{c}"] = (df[col] == c).astype(int)
    return X.reset_index(drop=True)
