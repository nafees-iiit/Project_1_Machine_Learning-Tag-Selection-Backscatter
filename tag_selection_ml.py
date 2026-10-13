"""
ML-Based Secure Tag Selection for Ambient Backscatter Communication
====================================================================
Author : Mohammad Nafees (Ph.D. Scholar, ECE, MNNIT Allahabad)

Problem
-------
An RF source S illuminates K backscatter tags. Each tag reflects S's signal
towards a legitimate reader D, while a passive eavesdropper E listens.
We must choose ONE tag so that the secrecy capacity is maximised.

The optimal rule (OTS) needs perfect channel knowledge of every link,
including the eavesdropper's links -- which is unrealistic, because a passive
eavesdropper does not transmit pilots. In practice we only have NOISY
estimates, and the eavesdropper estimates are much worse than the
legitimate-link estimates.

Idea
----
Train machine-learning classifiers (Random Forest and a Neural Network)
that take the NOISY channel estimates as input and predict the tag that is
truly best. Compare them with classical schemes using Secrecy Outage
Probability (SOP) obtained by Monte Carlo simulation.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures without a screen
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------------------------------
# 1. System parameters
# ------------------------------------------------------------------
SEED       = 42          # fixed seed -> same results every run
K          = 4           # number of backscatter tags
M_FADING   = 2           # Nakagami-m parameter for all links
OMEGA_MAIN = 1.0         # average power gain: S->T and T->D links
OMEGA_EVE  = 0.5         # average power gain: T->E (eavesdropper is farther)
BETA       = 0.5         # backscatter reflection coefficient of each tag
RS         = 0.5         # target secrecy rate (bits/s/Hz)
SIGMA_MAIN = 0.10        # channel-estimation error std, legitimate links
SIGMA_EVE  = 0.60        # channel-estimation error std, eavesdropper links
SNR_DB     = np.arange(0, 32, 2)   # transmit SNR grid (dB)
N_TRAIN_PER_SNR = 4000   # training samples per SNR point
N_TEST_PER_SNR  = 20000  # Monte Carlo test samples per SNR point

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------
# 2. Channel generation
# ------------------------------------------------------------------
def nakagami_complex(n, k, m, omega):
    """Complex channel coefficients whose power |h|^2 ~ Gamma(m, omega/m)
    (i.e. Nakagami-m fading) with a uniformly random phase."""
    power = rng.gamma(shape=m, scale=omega / m, size=(n, k))
    phase = rng.uniform(0, 2 * np.pi, size=(n, k))
    return np.sqrt(power) * np.exp(1j * phase)


def estimate(h, sigma):
    """Noisy channel estimate: h_hat = h + error, error ~ CN(0, sigma^2)."""
    err = (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)) * sigma / np.sqrt(2)
    return h + err


def generate_snapshots(n, snr_db):
    """Generate n independent channel snapshots at one SNR.
    Returns true power gains, estimated power gains and the SNR (linear)."""
    h = nakagami_complex(n, K, M_FADING, OMEGA_MAIN)   # S -> tag k
    g = nakagami_complex(n, K, M_FADING, OMEGA_MAIN)   # tag k -> reader D
    f = nakagami_complex(n, K, M_FADING, OMEGA_EVE)    # tag k -> eavesdropper E
    true = {"h": np.abs(h) ** 2, "g": np.abs(g) ** 2, "f": np.abs(f) ** 2}
    est = {"h": np.abs(estimate(h, SIGMA_MAIN)) ** 2,
           "g": np.abs(estimate(g, SIGMA_MAIN)) ** 2,
           "f": np.abs(estimate(f, SIGMA_EVE)) ** 2}
    return true, est, 10 ** (snr_db / 10)


# ------------------------------------------------------------------
# 3. Secrecy capacity of every tag (dyadic backscatter channel)
# ------------------------------------------------------------------
def secrecy_capacity(gains, snr):
    """C_s,k = [log2(1+SNR_D,k) - log2(1+SNR_E,k)]^+ for every tag k.
    SNR_D,k = snr * beta * |h_k|^2 * |g_k|^2   (S -> tag -> reader)
    SNR_E,k = snr * beta * |h_k|^2 * |f_k|^2   (S -> tag -> eavesdropper)"""
    snr_d = snr * BETA * gains["h"] * gains["g"]
    snr_e = snr * BETA * gains["h"] * gains["f"]
    return np.maximum(np.log2(1 + snr_d) - np.log2(1 + snr_e), 0.0)


# ------------------------------------------------------------------
# 4. Features for the ML model (what the ML model is allowed to see)
# ------------------------------------------------------------------
def make_features(est, snr_db):
    """Features = estimated gains of all 3K links in dB + SNR in dB.
    dB scale makes very small and very large gains comparable."""
    to_db = lambda x: 10 * np.log10(x + 1e-12)
    n = est["h"].shape[0]
    return np.hstack([to_db(est["h"]), to_db(est["g"]), to_db(est["f"]),
                      np.full((n, 1), snr_db)])


# ------------------------------------------------------------------
# 5. Build the training dataset (labels come from PERFECT channels)
# ------------------------------------------------------------------
X_list, y_list = [], []
for snr_db in SNR_DB:
    true, est, snr = generate_snapshots(N_TRAIN_PER_SNR, snr_db)
    X_list.append(make_features(est, snr_db))
    y_list.append(np.argmax(secrecy_capacity(true, snr), axis=1))  # truly best tag
X = np.vstack(X_list)
y = np.concatenate(y_list)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                            random_state=SEED, stratify=y)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {K} classes")
print(f"Train = {len(y_tr)}, Validation = {len(y_val)}")

# ------------------------------------------------------------------
# 6. Train two ML models
# ------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=150, max_depth=18, n_jobs=-1,
                            random_state=SEED)
mlp = make_pipeline(StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300,
                                  early_stopping=True, random_state=SEED))
models = {"Random Forest": rf, "Neural Network (MLP)": mlp}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_val, model.predict(X_val))
    print(f"{name:22s} trained in {time.time()-t0:5.1f} s | validation accuracy = {acc*100:5.2f} %")

# Baseline accuracy of the classical plug-in rule on the same validation data
cs_plugin_val = None   # computed per SNR below for SOP; accuracy shown at the end

# ------------------------------------------------------------------
# 7. Monte Carlo test: Secrecy Outage Probability vs SNR
# ------------------------------------------------------------------
schemes = ["OTS (perfect CSI, genie)", "Plug-in OTS (noisy CSI)",
           "SOTS (main link only)", "RTS (random)",
           "ML: Random Forest", "ML: Neural Network"]
sop = {s: [] for s in schemes}
acc_rows = []

for snr_db in SNR_DB:
    true, est, snr = generate_snapshots(N_TEST_PER_SNR, snr_db)
    cs_true = secrecy_capacity(true, snr)          # what really happens
    cs_est = secrecy_capacity(est, snr)            # what we can compute from estimates
    idx = np.arange(N_TEST_PER_SNR)
    feats = make_features(est, snr_db)
    best = np.argmax(cs_true, axis=1)

    choice = {
        "OTS (perfect CSI, genie)": best,
        "Plug-in OTS (noisy CSI)": np.argmax(cs_est, axis=1),
        "SOTS (main link only)": np.argmax(est["h"] * est["g"], axis=1),
        "RTS (random)": rng.integers(0, K, N_TEST_PER_SNR),
        "ML: Random Forest": rf.predict(feats),
        "ML: Neural Network": mlp.predict(feats),
    }
    row = {"SNR_dB": snr_db}
    for s in schemes:
        cs_sel = cs_true[idx, choice[s]]           # true capacity of chosen tag
        sop[s].append(np.mean(cs_sel < RS))        # outage if below target rate
        row[s] = np.mean(choice[s] == best)        # how often the truly best tag was picked
    acc_rows.append(row)

sop_df = pd.DataFrame({"SNR_dB": SNR_DB, **sop})
acc_df = pd.DataFrame(acc_rows)
sop_df.to_csv("results/sop_results.csv", index=False)
acc_df.to_csv("results/selection_accuracy.csv", index=False)

print("\nSecrecy Outage Probability (lower is better):")
print(sop_df.iloc[::3].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ------------------------------------------------------------------
# 8. Inference time (how fast each model decides)
# ------------------------------------------------------------------
feats = make_features(generate_snapshots(10000, 20)[1], 20)
for name, model in models.items():
    t0 = time.time(); model.predict(feats)
    print(f"{name:22s}: {1e6*(time.time()-t0)/10000:6.2f} microseconds per decision")

# ------------------------------------------------------------------
# 9. Figures
# ------------------------------------------------------------------
styles = {"OTS (perfect CSI, genie)": ("k", "-", "o"),
          "Plug-in OTS (noisy CSI)": ("tab:orange", "--", "s"),
          "SOTS (main link only)": ("tab:green", "--", "^"),
          "RTS (random)": ("tab:gray", ":", "v"),
          "ML: Random Forest": ("tab:blue", "-", "D"),
          "ML: Neural Network": ("tab:red", "-", "x")}
plt.figure(figsize=(7.5, 5))
for s in schemes:
    c, ls, mk = styles[s]
    plt.semilogy(SNR_DB, np.maximum(sop[s], 1e-5), color=c, linestyle=ls, marker=mk, label=s)
plt.xlabel("Transmit SNR (dB)"); plt.ylabel("Secrecy Outage Probability")
plt.title(f"SOP vs SNR (K={K} tags, Nakagami m={M_FADING}, R_s={RS})")
plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/sop_vs_snr.png", dpi=200); plt.close()

plt.figure(figsize=(7.5, 5))
for s in schemes:
    c, ls, mk = styles[s]
    plt.plot(SNR_DB, 100 * acc_df[s], color=c, linestyle=ls, marker=mk, label=s)
plt.xlabel("Transmit SNR (dB)"); plt.ylabel("Best tag correctly chosen (%)")
plt.title("How often each scheme picks the truly best tag")
plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/selection_accuracy.png", dpi=200); plt.close()

true, est, snr = generate_snapshots(N_TEST_PER_SNR, 20)
cm = confusion_matrix(np.argmax(secrecy_capacity(true, snr), axis=1),
                      rf.predict(make_features(est, 20)), normalize="true")
plt.figure(figsize=(5, 4.3))
plt.imshow(cm, cmap="Blues", vmin=0, vmax=1); plt.colorbar()
for i in range(K):
    for j in range(K):
        plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                 color="white" if cm[i, j] > 0.5 else "black")
plt.xticks(range(K), [f"Tag {k+1}" for k in range(K)])
plt.yticks(range(K), [f"Tag {k+1}" for k in range(K)])
plt.xlabel("Predicted best tag"); plt.ylabel("True best tag")
plt.title("Random Forest confusion matrix (SNR = 20 dB)")
plt.tight_layout(); plt.savefig("results/confusion_matrix_rf.png", dpi=200); plt.close()

names = [f"h{k+1}" for k in range(K)] + [f"g{k+1}" for k in range(K)] + \
        [f"f{k+1}" for k in range(K)] + ["SNR"]
imp = rf.feature_importances_
plt.figure(figsize=(7.5, 4))
plt.bar(names, imp, color=["tab:blue"] * K + ["tab:green"] * K + ["tab:red"] * K + ["gray"])
plt.ylabel("Importance"); plt.title("Random Forest feature importance (h: S-tag, g: tag-reader, f: tag-eavesdropper)")
plt.tight_layout(); plt.savefig("results/feature_importance.png", dpi=200); plt.close()

print("\nAll results saved in the 'results' folder.")
