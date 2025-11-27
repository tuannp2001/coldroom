# dual_branch_deltas_dkf3_nniSS.py
import numpy as np
import tensorflow as tf
from keras import layers, models

# =========================
# Config
# =========================
DATA_DIR    = "../data_coldroom/data_train0"
DATA_DIR1   = "../data_coldroom/data_train1"

SEQ_LEN     = 8
EPOCHS      = 100
LR          = 1e-3
BATCH_MAX   = 256

# Reasonable scales limiting residual jumps (tune as needed)
SCALE_A1 = 0.05
SCALE_A2 = 0.1
SCALE_A3 = 0.05


# =========================
# Load series
# =========================
Ta     = np.load(f"{DATA_DIR}/inside_air_denoised.npy").astype(np.float32)   # [N]
Tw     = np.load(f"{DATA_DIR}/inside_water_denoised.npy").astype(np.float32) # [N]
Ts     = np.load(f"{DATA_DIR}/T_supply_denoised.npy").astype(np.float32)     # [N]
status = np.load(f"{DATA_DIR}/status.npy").astype(np.float32)                # [N]
var_trend = np.load(f"{DATA_DIR}/var_profile_Ta.npy").astype(np.float32)     # [N]

Ta0     = np.load(f"{DATA_DIR1}/inside_air_denoised.npy").astype(np.float32)   # [N]
Tw0     = np.load(f"{DATA_DIR1}/inside_water_denoised.npy").astype(np.float32) # [N]
Ts0     = np.load(f"{DATA_DIR1}/T_supply_denoised.npy").astype(np.float32)     # [N]
status0 = np.load(f"{DATA_DIR1}/status.npy").astype(np.float32)                # [N]
var_trend0 = np.load(f"{DATA_DIR1}/var_profile_Ta.npy").astype(np.float32)     # [N]

def strict_checks(*arrs):
    L = [len(a) for a in arrs]
    if len(set(L)) != 1:
        raise ValueError(f"Lengths mismatch: {L}")
    for nm, a in zip(["Ta","Tw","Ts","status"], arrs):
        if not np.isfinite(a).all():
            raise ValueError(f"{nm} has NaN/Inf")
strict_checks(Ta, Tw, Ts, status)
strict_checks(Ta0, Tw0, Ts0, status0)

# ============ NEW: time_since_on/off ============

def time_since_last_status(s, target):
    """
    s: 1D array (0/1 float)
    target: 0 or 1
    return: the most recent consecutive steps equal to the target (up to the present time)
    """
    T = len(s)
    out = np.zeros(T, dtype=np.float32)
    run = 0
    for i in range(T):
        if (s[i] > 0.5) == bool(target):
            run += 1
        else:
            run = 0
        out[i] = run
    return out

ts_on     = time_since_last_status(status, 1)   # run 0
ts_off    = time_since_last_status(status, 0)
ts_on0    = time_since_last_status(status0, 1)  # run 1
ts_off0   = time_since_last_status(status0, 0)



# =========================
# Targets (LEVELS at t+1)
# =========================
Y  = np.column_stack([Ta[1:],  Tw[1:]]).astype(np.float32)
Y0 = np.column_stack([Ta0[1:], Tw0[1:]]).astype(np.float32)



# =========================
# Windowing helpers
# =========================
def make_seq_windows(f, L=SEQ_LEN):
    M = len(f)
    xs = [f[i-L+1:i+1] for i in range(L-1, M)]
    return np.stack(xs, axis=0)[:, :, None].astype(np.float32)

def make_target_windows(Y, L=SEQ_LEN):
    M = len(Y)
    ys = [Y[i] for i in range(L-1, M)]
    return np.stack(ys, axis=0).astype(np.float32)

# =========================
# Build tensors (inputs are raw levels at time t for transition t->t+1)
# =========================

# --- Run 0 (DATA_DIR) ---
X_Ta   = make_seq_windows(Ta[:-1],      SEQ_LEN)
X_Tw   = make_seq_windows(Tw[:-1],      SEQ_LEN)
X_Ts   = make_seq_windows(Ts[:-1],      SEQ_LEN)
X_s    = make_seq_windows(status[1:],   SEQ_LEN)
X_var  = make_seq_windows(var_trend[:-1], SEQ_LEN)          

# NEW: time_since_on/off windows (align with status[1:])
X_ton  = make_seq_windows(ts_on[1:],    SEQ_LEN)
X_toff = make_seq_windows(ts_off[1:],   SEQ_LEN)

Y_tg   = make_target_windows(Y, SEQ_LEN)

# --- Run 1 (DATA_DIR1) ---
X_Ta0   = make_seq_windows(Ta0[:-1],      SEQ_LEN)
X_Tw0   = make_seq_windows(Tw0[:-1],      SEQ_LEN)
X_Ts0   = make_seq_windows(Ts0[:-1],      SEQ_LEN)
X_s0    = make_seq_windows(status0[1:],   SEQ_LEN)
X_var0  = make_seq_windows(var_trend0[:-1], SEQ_LEN)

X_ton0  = make_seq_windows(ts_on0[1:],    SEQ_LEN)
X_toff0 = make_seq_windows(ts_off0[1:],   SEQ_LEN)

Y_tg0   = make_target_windows(Y0, SEQ_LEN)

# --- Concatenate both runs ---
X_Ta_all    = np.concatenate([X_Ta,   X_Ta0],   axis=0)
X_Tw_all    = np.concatenate([X_Tw,   X_Tw0],   axis=0)
X_Ts_all    = np.concatenate([X_Ts,   X_Ts0],   axis=0)
X_s_all     = np.concatenate([X_s,    X_s0],    axis=0)
X_var_all   = np.concatenate([X_var,  X_var0],  axis=0)

X_ton_all   = np.concatenate([X_ton,  X_ton0],  axis=0)   # NEW
X_toff_all  = np.concatenate([X_toff, X_toff0], axis=0)   # NEW

Y_tg_all    = np.concatenate([Y_tg,   Y_tg0],   axis=0)

B = len(Y_tg_all)
print(f"Windowed samples (run0 + run1): {B} with SEQ_LEN={SEQ_LEN}")

# =========================
# Targets: increments dTa, dTw at t -> t+1
# =========================

# --- Run 0 (DATA_DIR) ---
dTa  = Ta[1:] - Ta[:-1]
dTw  = Tw[1:] - Tw[:-1]
D    = np.column_stack([dTa, dTw]).astype(np.float32)   # [N-1, 2]
D_tg = make_target_windows(D, SEQ_LEN)                 # [B0, 2]

# --- Run 1 (DATA_DIR1) ---
dTa0  = Ta0[1:] - Ta0[:-1]
dTw0  = Tw0[1:] - Tw0[:-1]
D0    = np.column_stack([dTa0, dTw0]).astype(np.float32)
D_tg0 = make_target_windows(D0, SEQ_LEN)                # [B1, 2]

# --- Concatenate both runs ---
D_tg_all = np.concatenate([D_tg, D_tg0], axis=0)        # [B, 2]

B = len(D_tg_all)
print(f"Windowed samples (run0 + run1): {B} with SEQ_LEN={SEQ_LEN}")


# =========================
# Custom layers
# =========================
try:
    from keras.saving import register_keras_serializable
except Exception:
    from keras.utils import register_keras_serializable

@register_keras_serializable(package="pinn")
class PhysicalLossLayer(layers.Layer):
    """
    Adding physical loss based on linear model:

      dTa_phys = a1_lin*(Tw - Ta) + a2_lin*(Ts - Ta)
      dTw_phys = a3_lin*(Ta - Tw)

    Input:
      [dTa_pred, dTw_pred, twmt_last, tsmt_last, tamt_last, gate]

    Output:
      [dTa_pred, dTw_pred] (For data loss calculation)
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
        dTa_pred, dTw_pred, twmt_last, tsmt_last, tamt_last, gate = inputs
        # gate: (B,1) -> (B,)
        gate = tf.squeeze(gate, axis=-1)

        # ON/OFF coefficients from linear model
        a1 = gate * self.a1_on + (1.0 - gate) * self.a1_off
        a2 = gate * self.a2_on + (1.0 - gate) * self.a2_off
        a3 = gate * self.a3_on + (1.0 - gate) * self.a3_off

        # dTa_phys, dTw_phys from physical model
        dTa_phys = a1 * twmt_last + a2 * tsmt_last
        dTw_phys = a3 * tamt_last

        # Adjusted (B,1) 
        dTa_phys = tf.expand_dims(dTa_phys, -1)
        dTw_phys = tf.expand_dims(dTw_phys, -1)

        # physical loss
        phys_res = tf.square(dTa_pred - dTa_phys) + tf.square(dTw_pred - dTw_phys)
        phys_loss = tf.reduce_mean(phys_res)
        self.add_loss(self.lambda_phys * phys_loss)

        # Return dTa_pred, dTw_pred for data loss
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


LV_MIN, LV_MAX = -6.0, 6.0          # clamp log-variance
STD_MIN, STD_MAX = 1e-3, 10.0       # clamp std for reparam
EPS = 1e-8

@register_keras_serializable(package="dkf")
class DKF1_Variational(layers.Layer):
    """
    Variational 1-D sequence denoiser.
    """
    def __init__(self, latent_dim=6, hidden=64, recon_weight=0.5, kl_weight=1e-2, **kwargs):
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
            shape=(self.Dz,),
            initializer="zeros",
            trainable=True
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
        y_seq, u_seq = inputs                    # y: (B,L,1), u: (B,L,1)
        B = tf.shape(y_seq)[0]; L = tf.shape(y_seq)[1]

        h_u = self.enc_u2(self.enc_u1(u_seq))    # (B,L, 2H)

        z_mu_prev = tf.tile(self.z0_mu[None, :], [B, 1])
        z_lv_prev = tf.tile(self.z0_logvar[None, :], [B, 1])
        z_prev    = self._reparam(z_mu_prev, z_lv_prev, training)

        yhat_ta = tf.TensorArray(y_seq.dtype, size=L)
        kl_ta   = tf.TensorArray(y_seq.dtype, size=L)
        rec_ta  = tf.TensorArray(y_seq.dtype, size=L)

        def step(t, mu_prev, lv_prev, z_prev_s, yhat_acc, kl_acc, rec_acc):
            y_t  = y_seq[:, t, :]                # (B,1)
            hu_t = h_u[:,  t, :]                 # (B,2H)

            hp   = self.prior(tf.concat([z_prev_s, hu_t], axis=-1))
            mu_p = self.prior_mu(hp)
            lv_p = tf.clip_by_value(self.prior_lv(hp), LV_MIN, LV_MAX)

            hq   = self.post(tf.concat([z_prev_s, y_t, hu_t], axis=-1))
            mu_q = self.post_mu(hq)
            lv_q = tf.clip_by_value(self.post_lv(hq), LV_MIN, LV_MAX)

            z_t  = self._reparam(mu_q, lv_q, training)
            yhat = self.dec_mean(self.dec(z_t))  # (B,1)

            kl   = self._kl_diag(mu_q, lv_q, mu_p, lv_p)            # (B,)
            rec  = tf.reduce_mean(tf.square(y_t - yhat), axis=-1)   # (B,)

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

        yhat = tf.transpose(yhat_ta.stack(), [1, 0, 2])   # (B,L,1)
        kl   = tf.transpose(kl_ta.stack(),   [1, 0])      # (B,L)
        rec  = tf.transpose(rec_ta.stack(),  [1, 0])      # (B,L)

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

# =========================
# Model
# =========================
# Inputs: raw Ta, Tw, Ts, status, time-since-on/off
inp_Ta    = layers.Input((SEQ_LEN,1), name="seq_Ta_raw")
inp_Tw    = layers.Input((SEQ_LEN,1), name="seq_Tw_raw")
inp_Ts    = layers.Input((SEQ_LEN,1), name="seq_Ts_raw")
inp_s     = layers.Input((SEQ_LEN,1), name="seq_status")
inp_ton   = layers.Input((SEQ_LEN,1), name="seq_t_since_on")   # NEW
inp_toff  = layers.Input((SEQ_LEN,1), name="seq_t_since_off")  # NEW
# inp_var = layers.Input((SEQ_LEN,1), name="seq_var_trend")    # not used

Ts_minus_Ta = layers.Subtract(name="Ts_minus_Ta")([inp_Ts, inp_Ta])  # (B,L,1)
Tw_minus_Ta = layers.Subtract(name="Tw_minus_Ta")([inp_Tw, inp_Ta])  # (B,L,1)
Ta_minus_Tw = layers.Subtract(name="Ta_minus_Tw")([inp_Ta, inp_Tw])  # (B,L,1)
# last step for physical loss
twmt_last = layers.Lambda(lambda x: x[:, -1, 0], name="twmt_last")(Tw_minus_Ta)
tsmt_last = layers.Lambda(lambda x: x[:, -1, 0], name="tsmt_last")(Ts_minus_Ta)
tamt_last = layers.Lambda(lambda x: x[:, -1, 0], name="tamt_last")(Ta_minus_Tw)


ts_minus_ta_at_t = layers.Lambda(
    lambda x: x[:, -1:, :], name="ts_minus_ta_at_t", output_shape=(1,1)
)(Ts_minus_Ta)

reg = tf.keras.regularizers.l2(0.0)

gate_last = layers.Lambda(
    lambda s: s[:, -1:, :], name="gate_last", output_shape=(1,1)
)(inp_s)
gate_pool = layers.Lambda(
    lambda z: tf.reduce_mean(z, axis=2), name="gate_pool", output_shape=(1,)
)(gate_last)
gate_bool = layers.Lambda(
    lambda z: tf.cast(z > 0.5, tf.float32), name="gate_bool", output_shape=(1,)
)(gate_pool)

g = layers.Identity(name="gate")(gate_bool)
one_minus_g = layers.Lambda(
    lambda z: 1.0 - z, name="one_minus_g", output_shape=(1,)
)(g)

def enc_block(x, name):
    x = layers.GRU(64, return_sequences=True, name=f"{name}_gru1")(x)
    x = layers.GRU(64, name=f"{name}_gru2")(x)
    return x

# ====== NEW: encode time-since-on/off once ======
ctx_ts_on  = enc_block(inp_ton,  "ts_on")
ctx_ts_off = enc_block(inp_toff, "ts_off")

# Base context for air branch
ctx_air_base = layers.Concatenate(name="ctx_air_base")([
    enc_block(Tw_minus_Ta, "twmt"),
    enc_block(Ts_minus_Ta, "tsmt"),
    enc_block(inp_s,        "stat_air"),
    ctx_ts_on,
    ctx_ts_off,
])

ctx_a1 = ctx_air_base
ctx_a2 = ctx_air_base

# Base context for water branch
ctx_wat = layers.Concatenate(name="ctx_wat_base")([
    enc_block(Ta_minus_Tw, "tamt"),
    enc_block(inp_s,       "stat_wat"),
])

# ========= NEW HEADS: dTa_on/off, dTw_on/off =========

# Air increment ON
h_air_on = layers.Dense(64, activation="gelu",
                        kernel_regularizer=reg, name="air_on_fc1")(ctx_air_base)
h_air_on = layers.Dense(32, activation="gelu",
                        kernel_regularizer=reg, name="air_on_fc2")(h_air_on)
dTa_on   = layers.Dense(1, name="dTa_on")(h_air_on)

# Air increment OFF
h_air_off = layers.Dense(64, activation="gelu",
                         kernel_regularizer=reg, name="air_off_fc1")(ctx_air_base)
h_air_off = layers.Dense(32, activation="gelu",
                         kernel_regularizer=reg, name="air_off_fc2")(h_air_off)
dTa_off   = layers.Dense(1, name="dTa_off")(h_air_off)

# Water increment ON
h_wat_on = layers.Dense(64, activation="gelu",
                        kernel_regularizer=reg, name="wat_on_fc1")(ctx_wat)
h_wat_on = layers.Dense(32, activation="gelu",
                        kernel_regularizer=reg, name="wat_on_fc2")(h_wat_on)
dTw_on   = layers.Dense(1, name="dTw_on")(h_wat_on)

# Water increment OFF
h_wat_off = layers.Dense(64, activation="gelu",
                         kernel_regularizer=reg, name="wat_off_fc1")(ctx_wat)
h_wat_off = layers.Dense(32, activation="gelu",
                         kernel_regularizer=reg, name="wat_off_fc2")(h_wat_off)
dTw_off   = layers.Dense(1, name="dTw_off")(h_wat_off)

def mix_on_off(on, off, name):
    on_part  = layers.Multiply(name=f"{name}_on_mul")([g, on])
    off_part = layers.Multiply(name=f"{name}_off_mul")([one_minus_g, off])
    return layers.Add(name=f"{name}_mix")([on_part, off_part])

# dTa_pred, dTw_pred after mixing ON/OFF
dTa_pred = mix_on_off(dTa_on, dTa_off, "dTa")
dTw_pred = mix_on_off(dTw_on, dTw_off, "dTw")


# =========================
# Init from linear models
# =========================
from pathlib import Path
import joblib

model_dir = Path("./models")
model_dir = Path("./models")

# --- linear model coef. for physical loss ---
try:
    model_air_on  = joblib.load(model_dir / "model_air_on.pkl")   # coef_ = [a1, a2]
    model_air_off = joblib.load(model_dir / "model_air_off.pkl")
    model_tw_on   = joblib.load(model_dir / "model_tw_on.pkl")    # coef_ = [a3]
    model_tw_off  = joblib.load(model_dir / "model_tw_off.pkl")

    air_on_inits   = (float(model_air_on.coef_[0]),
                      float(model_air_on.coef_[1]))
    air_off_inits  = (float(model_air_off.coef_[0]),
                      float(model_air_off.coef_[1]))
    water_on_inits = (float(model_tw_on.coef_[0]),)
    water_off_inits= (float(model_tw_off.coef_[0]),)

    print("Loaded linear models for PINN physical loss.")
except Exception as e:
    print("WARNING: cannot load linear models, using zeros for PINN:", e)
    air_on_inits = air_off_inits = (0.0, 0.0)
    water_on_inits = water_off_inits = (0.0,)

phys_layer = PhysicalLossLayer(
    a1_on=air_on_inits[0],
    a2_on=air_on_inits[1],
    a1_off=air_off_inits[0],
    a2_off=air_off_inits[1],
    a3_on=water_on_inits[0],
    a3_off=water_off_inits[0],
    lambda_phys=0.3,            # tune: 0.1, 1.0, 10.0...
    name="pinn_phys_loss"
)

# áp physical loss lên dTa_pred, dTw_pred
dTa_phys_out, dTw_phys_out = phys_layer(
    [dTa_pred, dTw_pred, twmt_last, tsmt_last, tamt_last, g]
)

# đặt tên output cuối cùng
dTa_out = layers.Activation("linear", name="dTa")(dTa_phys_out)
dTw_out = layers.Activation("linear", name="dTw")(dTw_phys_out)


model = models.Model(
    inputs=[inp_Ta, inp_Tw, inp_Ts, inp_s, inp_ton, inp_toff],
    outputs=[dTa_out, dTw_out],
    name="DualBranch_PINN_deltas_withDur"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR, clipnorm=1.0),
    loss={"dTa": "mse", "dTw": "mse"},
    loss_weights={"dTa": 1.0, "dTw": 1.0},
    metrics={"dTa": ["mse"], "dTw": ["mse"]},
)


# ========= Validation set =========
DATA_DIR_VAL = "./data_validation"
Ta_val = np.load(f"{DATA_DIR_VAL}/inside_air_denoised.npy").astype(np.float32)
Tw_val = np.load(f"{DATA_DIR_VAL}/inside_water_denoised.npy").astype(np.float32)
Ts_val = np.load(f"{DATA_DIR_VAL}/T_supply_denoised.npy").astype(np.float32)
s_val  = np.load(f"{DATA_DIR_VAL}/status.npy").astype(np.float32)
var_trend_val = np.load(f"{DATA_DIR_VAL}/var_profile_Ta.npy").astype(np.float32)

# time-since-on/off cho validation
ts_on_val  = time_since_last_status(s_val, 1)
ts_off_val = time_since_last_status(s_val, 0)

Y_val   = np.column_stack([Ta_val[1:], Tw_val[1:]]).astype(np.float32)

dTa_val = Ta_val[1:] - Ta_val[:-1]
dTw_val = Tw_val[1:] - Tw_val[:-1]
D_val   = np.column_stack([dTa_val, dTw_val]).astype(np.float32)
D_val_tg = make_target_windows(D_val, SEQ_LEN)    # [B_val, 2]

D_val_Ta = D_val_tg[:, 0:1]
D_val_Tw = D_val_tg[:, 1:2]

X_Ta_val   = make_seq_windows(Ta_val[:-1], SEQ_LEN)
X_Tw_val   = make_seq_windows(Tw_val[:-1], SEQ_LEN)
X_Ts_val   = make_seq_windows(Ts_val[:-1], SEQ_LEN)
X_s_val    = make_seq_windows(s_val[1:],   SEQ_LEN)
X_var_val  = make_seq_windows(var_trend_val[:-1], SEQ_LEN)

X_ton_val  = make_seq_windows(ts_on_val[1:],  SEQ_LEN)
X_toff_val = make_seq_windows(ts_off_val[1:], SEQ_LEN)

Y_val_Ta = make_target_windows(Y_val, SEQ_LEN)[:, 0:1]
Y_val_Tw = make_target_windows(Y_val, SEQ_LEN)[:, 1:2]

# — Callbacks that monitor validation —
cbs = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_on_val_nniSS.keras", monitor="val_loss",
        save_best_only=True, save_weights_only=False, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=15, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.CSVLogger("training_nniSS_with_val.csv", append=False),
]

# =========================
# Train
# =========================
history = model.fit(
    x={
        "seq_Ta_raw":      X_Ta_all,
        "seq_Tw_raw":      X_Tw_all,
        "seq_Ts_raw":      X_Ts_all,
        "seq_status":      X_s_all,
        "seq_t_since_on":  X_ton_all,
        "seq_t_since_off": X_toff_all,
    },
    y={
        "dTa": D_tg_all[:, 0:1],
        "dTw": D_tg_all[:, 1:2],
    },
    validation_data=(
        {
            "seq_Ta_raw":      X_Ta_val,
            "seq_Tw_raw":      X_Tw_val,
            "seq_Ts_raw":      X_Ts_val,
            "seq_status":      X_s_val,
            "seq_t_since_on":  X_ton_val,
            "seq_t_since_off": X_toff_val,
        },
        {"dTa": D_val_Ta, "dTw": D_val_Tw}
    ),
    epochs=EPOCHS,
    batch_size=BATCH_MAX,
    verbose=1,
    callbacks=cbs,
    shuffle=True,
)

# Build once with zeros (ensures all sublayers are built before saving)
B0, L0 = 2, SEQ_LEN
dummy = {
    "seq_Ta_raw":      np.zeros((B0, L0, 1), np.float32),
    "seq_Tw_raw":      np.zeros((B0, L0, 1), np.float32),
    "seq_Ts_raw":      np.zeros((B0, L0, 1), np.float32),
    "seq_status":      np.zeros((B0, L0, 1), np.float32),
    "seq_t_since_on":  np.zeros((B0, L0, 1), np.float32),
    "seq_t_since_off": np.zeros((B0, L0, 1), np.float32),
}

_ = model(dummy)

model.save("dual_branch_deltas_nniSS.keras")
print("Saved to dual_branch_deltas_nniSS.keras")
