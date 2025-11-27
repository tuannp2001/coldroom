import numpy as np
from keras import models, layers
import matplotlib.pyplot as plt
import builtins, tensorflow as tf
builtins.tf = tf  # allow saved Lambda layers to see 'tf'

# ========= Config =========
MODEL_PATH = "dual_branch_deltas_nniSS.keras"
SEQ_LEN    = 8
WARM_STEPS = SEQ_LEN          # number of warm-up steps (ground truth levels)

# ========= Keras serializables matching TRAINING =========
try:
    from keras.saving import register_keras_serializable
except Exception:
    from keras.utils import register_keras_serializable

LV_MIN, LV_MAX = -6.0, 6.0
STD_MIN, STD_MAX = 1e-3, 10.0
EPS = 1e-8

@register_keras_serializable(package="dkf")
class DKF1_Variational(layers.Layer):
    def __init__(self, latent_dim=6, hidden=64,
                 recon_weight=0.5, kl_weight=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.Dy, self.Dz, self.H = 1, int(latent_dim), int(hidden)
        self.rw, self.kw = float(recon_weight), float(kl_weight)

        self.enc_u1 = layers.Bidirectional(
            layers.GRU(self.H, return_sequences=True),
            name=self.name+"_u_bigru1"
        )
        self.enc_u2 = layers.Bidirectional(
            layers.GRU(self.H, return_sequences=True),
            name=self.name+"_u_bigru2"
        )

        self.prior = tf.keras.Sequential([
            layers.Dense(self.H, activation="gelu"),
            layers.Dense(self.H, activation="gelu")
        ])
        self.prior_mu = layers.Dense(self.Dz)
        self.prior_lv = layers.Dense(self.Dz)

        self.post  = tf.keras.Sequential([
            layers.Dense(self.H, activation="gelu"),
            layers.Dense(self.H, activation="gelu")
        ])
        self.post_mu = layers.Dense(self.Dz)
        self.post_lv = layers.Dense(self.Dz)

        self.dec   = tf.keras.Sequential([
            layers.Dense(self.H, activation="gelu"),
            layers.Dense(self.H, activation="gelu")
        ])
        self.dec_mean = layers.Dense(self.Dy)

    def build(self, _):
        self.z0_mu     = self.add_weight(
            name=self.name+"_z0_mu",
            shape=(self.Dz,), initializer="zeros", trainable=True
        )
        self.z0_logvar = self.add_weight(
            name=self.name+"_z0_logvar",
            shape=(self.Dz,),
            initializer=tf.keras.initializers.Constant(-4.0),
            trainable=True
        )
        super().build(_)

    @staticmethod
    def _reparam(mu, lv, training):
        if not training:
            return mu
        lv  = tf.clip_by_value(lv, LV_MIN, LV_MAX)
        std = tf.exp(0.5 * lv)
        std = tf.clip_by_value(std, STD_MIN, STD_MAX)
        return mu + std * tf.random.normal(tf.shape(mu))

    @staticmethod
    def _kl_diag(mu_q, lv_q, mu_p, lv_p):
        lv_q = tf.clip_by_value(lv_q, LV_MIN, LV_MAX)
        lv_p = tf.clip_by_value(lv_p, LV_MIN, LV_MAX)
        var_q = tf.exp(lv_q)
        var_p = tf.exp(lv_p)
        return 0.5 * tf.reduce_sum(
            lv_p - lv_q + (var_q + tf.square(mu_q - mu_p)) / (var_p + EPS) - 1.0,
            axis=-1
        )

    def call(self, inputs, training=None):
        y_seq, u_seq = inputs
        B = tf.shape(y_seq)[0]; L = tf.shape(y_seq)[1]

        h_u = self.enc_u2(self.enc_u1(u_seq))

        z_mu_prev = tf.tile(self.z0_mu[None, :], [B, 1])
        z_lv_prev = tf.tile(self.z0_logvar[None, :], [B, 1])
        z_prev    = self._reparam(z_mu_prev, z_lv_prev, training)

        yhat_ta = tf.TensorArray(y_seq.dtype, size=L)
        kl_ta   = tf.TensorArray(y_seq.dtype, size=L)
        rec_ta  = tf.TensorArray(y_seq.dtype, size=L)

        def step(t, mu_prev, lv_prev, z_prev_s, yhat_acc, kl_acc, rec_acc):
            y_t  = y_seq[:, t, :]
            hu_t = h_u[:,  t, :]

            hp   = self.prior(tf.concat([z_prev_s, hu_t], axis=-1))
            mu_p = self.prior_mu(hp)
            lv_p = tf.clip_by_value(self.prior_lv(hp), LV_MIN, LV_MAX)

            hq   = self.post(tf.concat([z_prev_s, y_t, hu_t], axis=-1))
            mu_q = self.post_mu(hq)
            lv_q = tf.clip_by_value(self.post_lv(hq), LV_MIN, LV_MAX)

            z_t  = self._reparam(mu_q, lv_q, training)
            yhat = self.dec_mean(self.dec(z_t))

            kl   = self._kl_diag(mu_q, lv_q, mu_p, lv_p)
            rec  = tf.reduce_mean(tf.square(y_t - yhat), axis=-1)

            yhat_acc = yhat_acc.write(t, yhat)
            kl_acc   = kl_acc.write(t, kl)
            rec_acc  = rec_acc.write(t, rec)
            return t+1, mu_q, lv_q, z_t, yhat_acc, kl_acc, rec_acc

        _, _, _, _, yhat_ta, kl_ta, rec_ta = tf.while_loop(
            lambda t, *_: t < L,
            step,
            [tf.constant(0), z_mu_prev, z_lv_prev, z_prev,
             yhat_ta, kl_ta, rec_ta],
            parallel_iterations=1
        )

        yhat = tf.transpose(yhat_ta.stack(), [1, 0, 2])
        kl   = tf.transpose(kl_ta.stack(),   [1, 0])
        rec  = tf.transpose(rec_ta.stack(),  [1, 0])

        kl_mean  = tf.reduce_mean(tf.reduce_mean(kl,  axis=1))
        rec_mean = tf.reduce_mean(tf.reduce_mean(rec, axis=1))
        self.add_loss(self.rw * rec_mean + self.kw * kl_mean)
        return yhat

    def get_config(self):
        return {
            "latent_dim": self.Dz,
            "hidden": self.H,
            "recon_weight": self.rw,
            "kl_weight": self.kw,
            **super().get_config()
        }

@register_keras_serializable(package="pinn")
class PhysicalLossLayer(layers.Layer):
    """
    Layer dùng trong training để add physical loss.
    Trong inference, nó chỉ pass-through dTa, dTw (loss không dùng nữa).
    """
    def __init__(self,
                 a1_on, a2_on, a1_off, a2_off,
                 a3_on, a3_off,
                 lambda_phys=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.a1_on = float(a1_on)
        self.a2_on = float(a2_on)
        self.a1_off = float(a1_off)
        self.a2_off = float(a2_off)
        self.a3_on = float(a3_on)
        self.a3_off = float(a3_off)
        self.lambda_phys = float(lambda_phys)

    def call(self, inputs, training=None):
        # training=False, since inference mode
        # inputs = [dTa_pred, dTw_pred, twmt_last, tsmt_last, tamt_last, gate]
        dTa_pred, dTw_pred, *_ = inputs
        return [dTa_pred, dTw_pred]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "a1_on": self.a1_on,
            "a2_on": self.a2_on,
            "a1_off": self.a1_off,
            "a2_off": self.a2_off,
            "a3_on": self.a3_on,
            "a3_off": self.a3_off,
            "lambda_phys": self.lambda_phys,
        })
        return cfg

# ========= Load data =========
DATA_DIR = "./data_validation"
Ta_true = np.load(f"{DATA_DIR}/inside_air_denoised.npy").astype(np.float32)
Tw_true = np.load(f"{DATA_DIR}/inside_water_denoised.npy").astype(np.float32)
Ts_true = np.load(f"{DATA_DIR}/T_supply_denoised.npy").astype(np.float32)
status  = np.load(f"{DATA_DIR}/status.npy").astype(np.float32)
var_trend_val = np.load(f"{DATA_DIR}/var_profile_Ta.npy").astype(np.float32)

N = len(Ta_true)
assert len(Tw_true)==N and len(Ts_true)==N and len(status)==N and len(var_trend_val)==N

# ========= Helper to compute time-since-on / time-since-off =========
def compute_time_since(status_arr: np.ndarray):
    N = len(status_arr)
    t_on  = np.zeros(N, dtype=np.float32)
    t_off = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if status_arr[i] > 0.5:
            t_on[i]  = t_on[i-1] + 1 if i > 0 else 1
            t_off[i] = 0.0
        else:
            t_off[i] = t_off[i-1] + 1 if i > 0 else 1
            t_on[i]  = 0.0
    return t_on, t_off

t_since_on, t_since_off = compute_time_since(status)

# ========= Load model (PINN: outputs dTa, dTw) =========
custom_objects = {
    "DKF1_Variational": DKF1_Variational,
    "PhysicalLossLayer": PhysicalLossLayer,
}
model = models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False,
    safe_mode=False
)

print("Inputs :", [t.name for t in model.inputs])
print("Outputs:", [t.name for t in model.outputs])

# ========= Helpers =========
def seq_last_L(arr, L):
    """Take last L points of 1D array arr, shape -> (1,L,1)."""
    return np.asarray(arr[-L:], np.float32).reshape(1, L, 1)

# ========= Free-run (autoregressive) – PINN style (dTa,dTw) =========
Ta_hat = np.zeros_like(Ta_true, dtype=np.float32)
Tw_hat = np.zeros_like(Tw_true, dtype=np.float32)
Ta_hat[:WARM_STEPS] = Ta_true[:WARM_STEPS]
Tw_hat[:WARM_STEPS] = Tw_true[:WARM_STEPS]

dTa_pred = np.full(N, np.nan, np.float32)
dTw_pred = np.full(N, np.nan, np.float32)

for t in range(WARM_STEPS-1, N-1):
    Ta_hist   = Ta_hat[:t+1]
    Tw_hist   = Tw_hat[:t+1]
    Ts_hist   = Ts_true[:t+1]
    s_hist    = status[:t+1]
    t_on_hist  = t_since_on[:t+1]
    t_off_hist = t_since_off[:t+1]

    x_Ta   = seq_last_L(Ta_hist,   SEQ_LEN)
    x_Tw   = seq_last_L(Tw_hist,   SEQ_LEN)
    x_Ts   = seq_last_L(Ts_hist,   SEQ_LEN)
    x_s    = seq_last_L(s_hist,    SEQ_LEN)
    x_ton  = seq_last_L(t_on_hist, SEQ_LEN)
    x_toff = seq_last_L(t_off_hist, SEQ_LEN)

    dTa_step, dTw_step = model.predict(
        {
            "seq_Ta_raw":       x_Ta,
            "seq_Tw_raw":       x_Tw,
            "seq_Ts_raw":       x_Ts,
            "seq_status":       x_s,
            "seq_t_since_on":   x_ton,
            "seq_t_since_off":  x_toff,
        },
        verbose=0
    )
    dTa_step = float(np.asarray(dTa_step).reshape(-1)[0])
    dTw_step = float(np.asarray(dTw_step).reshape(-1)[0])

    dTa_pred[t+1] = dTa_step
    dTw_pred[t+1] = dTw_step

    # Free-run update bằng increment
    Ta_hat[t+1] = Ta_hat[t] + dTa_step
    Tw_hat[t+1] = Tw_hat[t] + dTw_step

# ========= Metrics =========
start = WARM_STEPS
rmse = lambda y,yh: float(np.sqrt(np.mean((y[start:] - yh[start:])**2)))
mae  = lambda y,yh: float(np.mean(np.abs(y[start:] - yh[start:])))

metrics = {
    "RMSE_Ta_free": rmse(Ta_true, Ta_hat),
    "MAE_Ta_free":  mae(Ta_true, Ta_hat),
    "RMSE_Tw_free": rmse(Tw_true, Tw_hat),
    "MAE_Tw_free":  mae(Tw_true, Tw_hat),
}
print(metrics)

# ========= Save =========
np.savez_compressed(
    "pinn_free_run.npz",
    Ta_hat=Ta_hat, Tw_hat=Tw_hat,
    dTa_pred=dTa_pred, dTw_pred=dTw_pred,
    Ta_true=Ta_true, Tw_true=Tw_true, Ts_true=Ts_true,
    status=status, var_trend=var_trend_val,
    t_since_on=t_since_on, t_since_off=t_since_off,
    seq_len=np.int32(SEQ_LEN), warm_steps=np.int32(WARM_STEPS),
    model_path=np.array(MODEL_PATH)
)

csv_mat = np.column_stack([
    Ta_true, Ta_hat, dTa_pred,
    Tw_true, Tw_hat, dTw_pred,
    Ts_true, status, var_trend_val,
    t_since_on, t_since_off
])
np.savetxt(
    "pinn_free_run.csv",
    csv_mat, delimiter=",",
    header="Ta_true,Ta_hat,dTa_pred,Tw_true,Tw_hat,dTw_pred,"
           "Ts_true,status,var_trend,t_since_on,t_since_off",
    comments=""
)

# ========= Plots =========
t_axis = np.arange(N)
on_mask = status.astype(bool)

fig, ax = plt.subplots(figsize=(11,4))
ax.plot(t_axis, Ta_true, label="Ta true", lw=1.5)
ax.plot(t_axis, Ta_hat,  label="Ta free-run (PINN)", lw=1.2)
ax.axvline(start, color="k", ls="--", alpha=0.5, label="free-run start")
ymin, ymax = ax.get_ylim()
ax.fill_between(t_axis, ymin, ymax, where=on_mask,
                alpha=0.08, step="pre", label="status=ON")
ax.set_ylim(ymin, ymax)
ax.set_title("Inside Air Temperature (Ta)")
ax.set_xlabel("t"); ax.set_ylabel("°C"); ax.legend(loc="best")
fig.tight_layout(); plt.savefig("pinn_free_run_Ta.png", dpi=140)
plt.show()

fig, ax = plt.subplots(figsize=(11,4))
ax.plot(t_axis, Tw_true, label="Tw true", lw=1.5)
ax.plot(t_axis, Tw_hat,  label="Tw free-run (PINN)", lw=1.2)
ax.axvline(start, color="k", ls="--", alpha=0.5, label="free-run start")
ymin, ymax = ax.get_ylim()
ax.fill_between(t_axis, ymin, ymax, where=on_mask,
                alpha=0.08, step="pre", label="status=ON")
ax.set_ylim(ymin, ymax)
ax.set_title("Water Temperature (Tw)")
ax.set_xlabel("t"); ax.set_ylabel("°C"); ax.legend(loc="best")
fig.tight_layout(); plt.savefig("pinn_free_run_Tw.png", dpi=140)
plt.show()
