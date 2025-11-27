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


@register_keras_serializable(package="nniSS")
class RegimeNNiSSResidual(layers.Layer):
    """
    SAME as in TRAINING: combine regimes, predict NEXT LEVELS:

        dTa = a1*(Tw - Ta) + a2*(Ts - Ta) + b_air
        dTw = a3*(Ta - Tw) + b_wat
        Ta_next = Ta_prev + dTa
        Tw_next = Tw_prev + dTw

    Inputs:
      [seq_tw_minus_ta, ts_minus_ta_at_t, seq_ta_minus_tw, seq_status,
       seq_Ta_raw, da1_in, da2_in, da3_in, db_air_in, db_wat_in]
    """
    def __init__(self,
                 init_air_on=(0., 0.),
                 init_air_off=(0., 0.),
                 init_water_on=(0.,),
                 init_water_off=(0.,),
                 scale_a1=0.05, scale_a2=0.1, scale_a3=0.05,
                 scale_b_air=0.05, scale_b_wat=0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.init_air_on   = tuple(float(x) for x in init_air_on)
        self.init_air_off  = tuple(float(x) for x in init_air_off)
        self.init_water_on = tuple(float(x) for x in init_water_on)
        self.init_water_off= tuple(float(x) for x in init_water_off)

        self.scale_a1 = float(scale_a1)
        self.scale_a2 = float(scale_a2)
        self.scale_a3 = float(scale_a3)
        self.scale_b_air = float(scale_b_air)
        self.scale_b_wat = float(scale_b_wat)

    def _scalar(self, name, val):
        return self.add_weight(
            name=name, shape=(),
            initializer=tf.keras.initializers.Constant(val),
            trainable=False
        )

    def build(self, _):
        # a1, a2 for ON/OFF
        self.a1_on  = self._scalar("a1_on",  self.init_air_on[0])
        self.a2_on  = self._scalar("a2_on",  self.init_air_on[1])
        self.a1_off = self._scalar("a1_off", self.init_air_off[0])
        self.a2_off = self._scalar("a2_off", self.init_air_off[1])

        # a3 for ON/OFF
        self.a3_on  = self._scalar("a3_on",  self.init_water_on[0])
        self.a3_off = self._scalar("a3_off", self.init_water_off[0])

        super().build(_)

    def call(self, inputs, training=None):
        (seq_tw_minus_ta,
         ts_minus_ta_at_t,
         seq_ta_minus_tw,
         seq_status,
         seq_Ta_raw,          # raw Ta sequence (B,L,1)
         da1_in, da2_in, da3_in,
         db_air_in, db_wat_in) = inputs

        # Last-step diffs
        twmt_last = seq_tw_minus_ta[:, -1, 0]  # (B,)
        tsmt_last = tf.reshape(ts_minus_ta_at_t,
                               [tf.shape(ts_minus_ta_at_t)[0], -1])[:, -1]  # (B,)
        tamt_last = seq_ta_minus_tw[:, -1, 0]  # (B,)

        # Previous levels
        Ta_prev = seq_Ta_raw[:, -1, 0]         # (B,)
        Tw_prev = Ta_prev + twmt_last          # vì Tw - Ta = twmt_last

        # Regime selector from last status in window
        s = tf.cast(seq_status[:, -1, 0] > 0.5, tf.float32)  # (B,)

        def pick(on_val, off_val):
            return on_val * s + off_val * (1.0 - s)

        # Initial params từ linear model (non-trainable)
        a1_init = pick(self.a1_on,  self.a1_off)   # (B,)
        a2_init = pick(self.a2_on,  self.a2_off)   # (B,)
        a3_init = pick(self.a3_on,  self.a3_off)   # (B,)

        # Residuals from heads (B,1) -> (B,)
        da1    = tf.squeeze(da1_in,  -1)
        da2    = tf.squeeze(da2_in,  -1)
        da3    = tf.squeeze(da3_in,  -1)
        db_air = tf.squeeze(db_air_in, -1)
        db_wat = tf.squeeze(db_wat_in, -1)

        # Smooth residuals
        a1 = a1_init + self.scale_a1 * tf.tanh(da1)
        a2 = a2_init + self.scale_a2 * tf.tanh(da2)
        a3 = a3_init + self.scale_a3 * tf.tanh(da3)

        b_air = self.scale_b_air * tf.tanh(db_air)
        b_wat = self.scale_b_wat * tf.tanh(db_wat)

        dTa_pred = a1 * twmt_last + a2 * tsmt_last + b_air
        dTw_pred = a3 * tamt_last + b_wat

        Ta_next = Ta_prev + dTa_pred
        Tw_next = Tw_prev + dTw_pred

        return [
            tf.expand_dims(Ta_next, -1),
            tf.expand_dims(Tw_next, -1)
        ]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "init_air_on": self.init_air_on,
            "init_air_off": self.init_air_off,
            "init_water_on": self.init_water_on,
            "init_water_off": self.init_water_off,
            "scale_a1": self.scale_a1,
            "scale_a2": self.scale_a2,
            "scale_a3": self.scale_a3,
            "scale_b_air": self.scale_b_air,
            "scale_b_wat": self.scale_b_wat,
        })
        return cfg


# ---- DKF1_Variational (giữ nguyên như training, nhưng có thể không dùng) ----
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
    """
    status_arr: (N,) in {0,1} (hoặc [0,1] float).
    Trả về:
      t_since_on[t]  = số bước liên tiếp status==1 tính đến t (kể cả t)
      t_since_off[t] = số bước liên tiếp status==0 tính đến t (kể cả t)
    """
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

# ========= Load model =========
custom_objects = {
    "RegimeNNiSSResidual": RegimeNNiSSResidual,
    "DKF1_Variational": DKF1_Variational,
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

# ========= Free-run (autoregressive) – NO noise injection =========
Ta_hat = np.zeros_like(Ta_true, dtype=np.float32)
Tw_hat = np.zeros_like(Tw_true, dtype=np.float32)
Ta_hat[:WARM_STEPS] = Ta_true[:WARM_STEPS]
Tw_hat[:WARM_STEPS] = Tw_true[:WARM_STEPS]

dTa_pred = np.full(N, np.nan, np.float32)
dTw_pred = np.full(N, np.nan, np.float32)

for t in range(WARM_STEPS-1, N-1):
    # lịch sử đến thời điểm t (KHÔNG thêm noise)
    Ta_hist   = Ta_hat[:t+1]
    Tw_hist   = Tw_hat[:t+1]
    Ts_hist   = Ts_true[:t+1]
    s_hist    = status[:t+1]
    t_on_hist  = t_since_on[:t+1]
    t_off_hist = t_since_off[:t+1]

    # lấy last SEQ_LEN làm input
    x_Ta   = seq_last_L(Ta_hist,   SEQ_LEN)
    x_Tw   = seq_last_L(Tw_hist,   SEQ_LEN)
    x_Ts   = seq_last_L(Ts_hist,   SEQ_LEN)
    x_s    = seq_last_L(s_hist,    SEQ_LEN)
    x_ton  = seq_last_L(t_on_hist, SEQ_LEN)
    x_toff = seq_last_L(t_off_hist, SEQ_LEN)

    # TÊN INPUT phải khớp với script TRAIN:
    #   Input(..., name="seq_t_since_on")
    #   Input(..., name="seq_t_since_off")
    Ta_next_pred, Tw_next_pred = model.predict(
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
    Ta_next_pred = float(np.asarray(Ta_next_pred).reshape(-1)[0])
    Tw_next_pred = float(np.asarray(Tw_next_pred).reshape(-1)[0])

    # Lưu delta để check
    dTa_pred[t+1] = Ta_next_pred - Ta_hat[t]
    dTw_pred[t+1] = Tw_next_pred - Tw_hat[t]

    # Free-run update bằng level không nhiễu
    Ta_hat[t+1] = Ta_next_pred
    Tw_hat[t+1] = Tw_next_pred

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
    "nniSS_free_run.npz",
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
    "nniSS_free_run.csv",
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
ax.plot(t_axis, Ta_hat,  label="Ta free-run", lw=1.2)
ax.axvline(start, color="k", ls="--", alpha=0.5, label="free-run start")
ymin, ymax = ax.get_ylim()
ax.fill_between(t_axis, ymin, ymax, where=on_mask,
                alpha=0.08, step="pre", label="status=ON")
ax.set_ylim(ymin, ymax)
ax.set_title("Inside Air Temperature (Ta)")
ax.set_xlabel("t"); ax.set_ylabel("°C"); ax.legend(loc="best")
fig.tight_layout(); plt.savefig("nniSS_free_run_Ta.png", dpi=140)
plt.show()

fig, ax = plt.subplots(figsize=(11,4))
ax.plot(t_axis, Tw_true, label="Tw true", lw=1.5)
ax.plot(t_axis, Tw_hat,  label="Tw free-run", lw=1.2)
ax.axvline(start, color="k", ls="--", alpha=0.5, label="free-run start")
ymin, ymax = ax.get_ylim()
ax.fill_between(t_axis, ymin, ymax, where=on_mask,
                alpha=0.08, step="pre", label="status=ON")
ax.set_ylim(ymin, ymax)
ax.set_title("Water Temperature (Tw)")
ax.set_xlabel("t"); ax.set_ylabel("°C"); ax.legend(loc="best")
fig.tight_layout(); plt.savefig("nniSS_free_run_Tw.png", dpi=140)
plt.show()
