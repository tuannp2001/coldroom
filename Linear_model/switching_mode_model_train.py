import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge


# ==== Flags ====
USE_OUTSIDE   = False


import numpy as np

def load_run(run_dir: str):
    """
    run_dir: e.g. 'data_train0', 'data_train1', 'data_validation'
    """
    base = f"./{run_dir}"
    Ta     = np.load(f"{base}/inside_air_denoised.npy")
    Tw     = np.load(f"{base}/inside_water_denoised.npy")
    To     = np.load(f"{base}/T_outside.npy")
    Ts     = np.load(f"{base}/T_supply_denoised.npy")
    Tr     = np.load(f"{base}/T_return_denoised.npy")
    status = np.load(f"{base}/status.npy")
    return Ta, Tw, To, Ts, Tr, status


# ==== Load training data: RUN 0 ====
Ta0, Tw0, To0, Ts0, Tr0, status0 = load_run("data_train0")

# ==== Load training data: RUN 1 ====
Ta1, Tw1, To1, Ts1, Tr1, status1 = load_run("data_train1")

# ==== Load testing / validation data ====
Ta_test, Tw_test, To_test, Ts_test, Tr_test, status_test = load_run("data_validation")


def strict_checks(Ta, Tw, Ts, Tr, To, status, use_outside=True):
    lengths = [len(Ta), len(Tw), len(Ts), len(status)]
    if use_outside:
        lengths.append(len(To))
    # you can also check Tr length if you want:
    lengths.append(len(Tr))

    if len(set(lengths)) != 1:
        raise ValueError(f"Series have different lengths: {lengths}")

    for name, arr in [
        ("Ta", Ta), ("Tw", Tw), ("Ts", Ts), ("Tr", Tr),
        ("To", To if use_outside else None), ("status", status)
    ]:
        if arr is None:
            continue
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} has NaN/Inf")


def make_pairs(Ta, Tw, Ts, Tr, To, status, use_outside=True):
    """
    Returns:
      X_air: shape [N-1, n_feat_air]
      X_tw : shape [N-1, n_feat_tw]
      Y    : shape [N-1, 2]  (ΔTa, ΔTw)
      stat : shape [N-1]     (aligned with deltas at t->t+1)
    """
    dTa = (Ta[1:] - Ta[:-1])
    dTw = (Tw[1:] - Tw[:-1])
    Y   = np.column_stack([dTa, dTw])

    X_air = np.column_stack([
        Tw[:-1] - Ta[:-1],
        Ts[:-1] - Ta[:-1],
        # if later you want To, you can add it here
        # (To[1:] - Ta[:-1]) if use_outside else 0.0
    ])

    X_tw = np.column_stack([
        Ta[:-1] - Tw[:-1],
    ])

    stat = status[1:]  # status for transition t -> t+1

    return X_air, X_tw, Y, stat

# ---- Build supervised pairs for RUN 0 ----
strict_checks(Ta0, Tw0, Ts0, Tr0, To0, status0, USE_OUTSIDE)
X_air_0, X_tw_0, Y_0, stat_0 = make_pairs(Ta0, Tw0, Ts0, Tr0, To0, status0, USE_OUTSIDE)

# ---- Build supervised pairs for RUN 1 ----
strict_checks(Ta1, Tw1, Ts1, Tr1, To1, status1, USE_OUTSIDE)
X_air_1, X_tw_1, Y_1, stat_1 = make_pairs(Ta1, Tw1, Ts1, Tr1, To1, status1, USE_OUTSIDE)

X_air_train   = np.concatenate([X_air_0, X_air_1], axis=0)
X_tw_train    = np.concatenate([X_tw_0,  X_tw_1],  axis=0)
Y_train       = np.concatenate([Y_0,      Y_1],     axis=0)
status_train  = np.concatenate([stat_0,   stat_1],  axis=0)

"""
X_air_train   = X_air_0
X_tw_train    = X_tw_0
Y_train       = Y_0
status_train  = stat_0
"""

m_on  = (status_train == 1)
m_off = ~m_on


print(X_air_train.shape, Ta_test.shape, X_tw_train .shape, Tw_test.shape)



RIDGE_ALPHA = 0.0
bias = False

model_air_on  = Ridge(alpha=RIDGE_ALPHA, fit_intercept=bias, positive=True).fit(
    X_air_train[m_on],  Y_train[m_on, 0]
)
model_air_off = Ridge(alpha=RIDGE_ALPHA, fit_intercept=bias, positive=True).fit(
    X_air_train[m_off], Y_train[m_off, 0]
)
model_tw_on   = Ridge(alpha=RIDGE_ALPHA, fit_intercept=bias, positive=True).fit(
    X_tw_train[m_on],   Y_train[m_on, 1]
)
model_tw_off  = Ridge(alpha=RIDGE_ALPHA, fit_intercept=bias, positive=True).fit(
    X_tw_train[m_off],  Y_train[m_off, 1]
)

plt.plot(Ta_test, label="Ta_test")
plt.plot(Ts_test, label="Ts_test")
plt.legend()
plt.show()


print([model_air_on.coef_, model_air_on.intercept_])
print([model_air_off.coef_, model_air_off.intercept_])
print([model_tw_on.coef_, model_tw_on.intercept_])
print([model_tw_off.coef_, model_tw_off.intercept_])



# ==== Save models ====
model_dir = Path("./models")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model_air_on,  model_dir / "model_air_on.pkl")
joblib.dump(model_air_off, model_dir / "model_air_off.pkl")
joblib.dump(model_tw_on,   model_dir / "model_tw_on.pkl")
joblib.dump(model_tw_off,  model_dir / "model_tw_off.pkl")
print(f"Models saved in {model_dir.resolve()}")
# ==== Free-run simulation on testing data and plotting ====

def simulate_free_run(
    Ta_init, Tw_init, Ts, status,
    model_air_on, model_air_off, model_tw_on, model_tw_off,
):
    """
    Free-run forward simulation from t -> t+1 using the trained models.
    - Inputs at time t: current predicted Ta_hat(t), Tw_hat(t), and Ts(t)
    - Mode selection uses status[t+1] (aligned with how we trained with stat = status[1:])
    - Outputs arrays have same length as inputs; index 0 equals the given initials.
    """
    N = len(Ts)
    Ta_hat = np.zeros(N)
    Tw_hat = np.zeros(N)
    Ta_hat[0] = float(Ta_init)
    Tw_hat[0] = float(Tw_init)

    for t in range(N - 1):
        # Features mirror training: [Tw(t) - Ta(t), Ts(t) - Ta(t)] for air; [Ta(t) - Tw(t)] for water
        x_air = np.array([[Tw_hat[t] - Ta_hat[t],Ts[t] - Ta_hat[t]]])
        x_tw  = np.array([[Ta_hat[t] - Tw_hat[t]]])

        # Mode used for transition t -> t+1
        on = (status[t + 1] == 1)
        # Predict deltas
        if on:
            dTa_pred = model_air_on.predict(x_air)[0]
            dTw_pred = model_tw_on.predict(x_tw)[0]
        else:
            dTa_pred = model_air_off.predict(x_air)[0]
            dTw_pred = model_tw_off.predict(x_tw)[0]

        # Update states
        Ta_hat[t + 1] = Ta_hat[t] + dTa_pred
        Tw_hat[t + 1] = Tw_hat[t] + dTw_pred

    return Ta_hat, Tw_hat

# Run free-run sim on testing set (init with the first observed values)
Ta_hat0, Tw_hat0 = simulate_free_run(
    Ta_init=Ta_test[0],
    Tw_init=Tw_test[0],
    Ts=Ts_test,
    status=status_test,
    model_air_on=model_air_on,
    model_air_off=model_air_off,
    model_tw_on=model_tw_on,
    model_tw_off=model_tw_off,
)

# ---- Build predicted delta series from the free-run states ----
dTa_pred_series = np.full_like(Ta_hat0, np.nan, dtype=np.float64)
dTw_pred_series = np.full_like(Tw_hat0, np.nan, dtype=np.float64)
dTa_pred_series[1:] = Ta_hat0[1:] - Ta_hat0[:-1]
dTw_pred_series[1:] = Tw_hat0[1:] - Tw_hat0[:-1]



# ==== Evaluation (RMSE from index 1 to align with first-step forecasting) ====
rmse_Ta = np.sqrt(mean_absolute_error(Ta_test[1:], Ta_hat0[1:]))
rmse_Tw = np.sqrt(mean_absolute_error(Tw_test[1:], Tw_hat0[1:]))

print(f"[TEST] RMSE Ta = {rmse_Ta:.4f} °C, RMSE Tw = {rmse_Tw:.4f} °C")


# ==== Save results (arrays + metadata) ====
np.savez_compressed(
    "lr_free_run_results.npz",
    Ta_true=Ta_test,
    Tw_true=Tw_test,
    Ts_true=Ts_test,
    status=status_test,
    Ta_hat=Ta_hat0,
    Tw_hat=Tw_hat0,
    dTa_pred=dTa_pred_series,
    dTw_pred=dTw_pred_series,
    rmse_Ta=np.float32(rmse_Ta),
    rmse_Tw=np.float32(rmse_Tw),
    ridge_alpha=np.float32(RIDGE_ALPHA),
    use_outside=np.int8(USE_OUTSIDE)
)
print("Saved arrays & metadata to lr_free_run_results.npz")

# Also a human-friendly CSV
csv_mat = np.column_stack([
    Ta_test, Ta_hat0, dTa_pred_series,
    Tw_test, Tw_hat0, dTw_pred_series,
    Ts_test, status_test
])
np.savetxt(
    "lr_free_run_results.csv",
    csv_mat,
    delimiter=",",
    header="Ta_true,Ta_hat,dTa_pred,Tw_true,Tw_hat,dTw_pred,Ts_true,status",
    comments=""
)
print("Saved table to lr_free_run_results.csv")


# ==== Plot ====
plt.figure(figsize=(12, 6))
plt.plot(Ta_test, label="Ta true", linewidth=1.5)
plt.plot(Ta_hat0, label="Ta pred (free-run)", linewidth=1.5, linestyle="--")
plt.plot(Tw_test, label="Tw true", linewidth=1.5)
plt.plot(Tw_hat0, label="Tw pred (free-run)", linewidth=1.5, linestyle="--")
plt.title("Free-run simulation on test data")
plt.xlabel("Time (10s)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

