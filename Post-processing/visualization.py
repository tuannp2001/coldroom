from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Load ground truth ----------
Ta_true = np.load("./data_validation/inside_air_denoised.npy")
Tw_true = np.load("./data_validation/inside_water_denoised.npy")

# ---------- Load predictions ----------
results_NNrSS    = pd.read_csv("nnrSS_free_run.csv")
results_NNISS = pd.read_csv("nniSS_free_run.csv")  # <<< NEW: NNiSS

Ta_pred_NNrSS     = results_NNrSS["Ta_hat"].to_numpy()
Tw_pred_NNrSS     = results_NNrSS["Tw_hat"].to_numpy()

Ta_pred_NNISS  = results_NNISS["Ta_hat"].to_numpy()   # <<< NEW: NNiSS
Tw_pred_NNISS  = results_NNISS["Tw_hat"].to_numpy()   # <<< NEW: NNiSS


# ---------- Align lengths (in case files differ slightly) ----------
series = [Ta_true, Tw_true,
          Ta_pred_NNrSS, Tw_pred_NNrSS,
          Ta_pred_NNISS, Tw_pred_NNISS]               # <<< NEW: NNiSS

L = min(map(len, series))

Ta_true       = Ta_true[:L]
Tw_true       = Tw_true[:L]
Ta_pred_NNrSS    = Ta_pred_NNrSS[:L]
Tw_pred_NNrSS    = Tw_pred_NNrSS[:L]

Ta_pred_NNISS = Ta_pred_NNISS[:L]                     # <<< NEW: NNiSS
Tw_pred_NNISS = Tw_pred_NNISS[:L]                     # <<< NEW: NNiSS

# ---------- Metric helper ----------
def metrics(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred),
            mean_absolute_error(y_true, y_pred))

# Compute metrics for each model
models = {
    "LR"     : (Ta_pred_NNrSS, Tw_pred_NNrSS),
    "NNiSS"  : (Ta_pred_NNISS, Tw_pred_NNISS),        # <<< NEW: NNiSS
}

print("\n=== Performance Comparison ===")
for name, (Ta_p, Tw_p) in models.items():
    mse_ta, mae_ta = metrics(Ta_true, Ta_p)
    mse_tw, mae_tw = metrics(Tw_true, Tw_p)
    print(f"{name:<7} Ta  MSE={mse_ta:.4f}  MAE={mae_ta:.4f} | "
          f"Tw  MSE={mse_tw:.4f}  MAE={mae_tw:.4f}")

# ---------- Plot ----------
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
t = np.arange(L)

# Air temperature
ax[0].plot(t, Ta_true,            label="Ta true",            color="black", linewidth=1.5)
ax[0].plot(t, Ta_pred_NNrSS,    "--", label="Ta LR",             linewidth=1.2)
ax[0].plot(t, Ta_pred_NNISS,       label="Ta NNrSS",          linewidth=1.2)  # <<< NEW: NNiSS
ax[0].set_ylabel("Temperature (°C)")
ax[0].set_title("Inside Air Temperature (Ta)")
ax[0].legend(ncol=2)

# Water temperature
ax[1].plot(t, Tw_true,            label="Tw true",            color="black", linewidth=1.5)
ax[1].plot(t, Tw_pred_NNrSS,    "--", label="Tw LR",             linewidth=1.2)
ax[1].plot(t, Tw_pred_NNISS,       label="Tw NNrSS",          linewidth=1.2)  # <<< NEW: NNiSS
ax[1].set_xlabel("Time step")
ax[1].set_ylabel("Temperature (°C)")
ax[1].set_title("Inside Water Temperature (Tw)")
ax[1].legend(ncol=2)

plt.tight_layout()
plt.show()

# ---------- Per-step errors (each instant) ----------
# Build a dict of predictions aligned with names
preds = {
    "LR":     (Ta_pred_NNrSS,    Tw_pred_NNrSS),
    "NNiSS":  (Ta_pred_NNISS, Tw_pred_NNISS),
}

# Prepare a dataframe with time index
t = np.arange(L)
cols = {"t": t, "Ta_true": Ta_true, "Tw_true": Tw_true}

# For each model, compute signed error, absolute error, and squared error at each time step
for name, (Ta_p, Tw_p) in preds.items():
    # Air
    ta_err      = Ta_p - Ta_true
    ta_abs_err  = np.abs(ta_err)
    ta_sq_err   = ta_err**2
    cols[f"Ta_err_{name}"]     = ta_err
    cols[f"Ta_abs_err_{name}"] = ta_abs_err
    cols[f"Ta_sq_err_{name}"]  = ta_sq_err

    # Water
    tw_err      = Tw_p - Tw_true
    tw_abs_err  = np.abs(tw_err)
    tw_sq_err   = tw_err**2
    cols[f"Tw_err_{name}"]     = tw_err
    cols[f"Tw_abs_err_{name}"] = tw_abs_err
    cols[f"Tw_sq_err_{name}"]  = tw_sq_err

# Assemble dataframe and save
per_step_df = pd.DataFrame(cols)
per_step_df.to_csv("per_step_errors.csv", index=False)
print("Saved per-step errors to per_step_errors.csv")
print(per_step_df.head())

# ----10---- Per-step errors with adjustable temporal resolution ----------
step =  100    # <<< compute/plot error every 5 time steps (1 = every point)010

t = np.arange(0, L, step)
cols = {"t": t, "Ta_true": Ta_true[::step], "Tw_true": Tw_true[::step]}

for name, (Ta_p, Tw_p) in preds.items():
    # Downsample predictions to chosen stride
    ta_p = Ta_p[::step]
    tw_p = Tw_p[::step]

    ta_err = ta_p - Ta_true[::step]
    cols[f"Ta_err_{name}"]     = ta_err
    cols[f"Ta_abs_err_{name}"] = np.abs(ta_err)
    cols[f"Ta_sq_err_{name}"]  = ta_err**2

    tw_err = tw_p - Tw_true[::step]
    cols[f"Tw_err_{name}"]     = tw_err
    cols[f"Tw_abs_err_{name}"] = np.abs(tw_err)
    cols[f"Tw_sq_err_{name}"]  = tw_err**2

per_step_df = pd.DataFrame(cols)
per_step_df.to_csv("per_step_errors_stride.csv", index=False)
print(f"Saved per-step errors every {step} time steps to per_step_errors_stride.csv")

# ---------- Plot downsampled absolute errors ----------
fig2, ax2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for name in preds.keys():
    ax2[0].plot(t, per_step_df[f"Ta_abs_err_{name}"], label=f"Ta | {name}")
ax2[0].set_title(f"|Ta| absolute error (every {step} steps)")
ax2[0].set_ylabel("|Error| (°C)")
ax2[0].legend(ncol=2)

for name in preds.keys():
    ax2[1].plot(t, per_step_df[f"Tw_abs_err_{name}"], label=f"Tw | {name}")
ax2[1].set_title(f"|Tw| absolute error (every {step} steps)")
ax2[1].set_xlabel("Time step")
ax2[1].set_ylabel("|Error| (°C)")
ax2[1].legend(ncol=2)

plt.tight_layout()
plt.show()
