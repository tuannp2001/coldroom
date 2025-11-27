import numpy as np
import matplotlib.pyplot as plt

# =========================
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


# =========================
# 2. Stats for each OFF segment
# =========================
def compute_off_segment_stats(signal: np.ndarray,
                              status: np.ndarray):
    """
    signal : 1D (Ta, Tw, ...)
    status : 1D, 0 = OFF, 1 = ON

    Returns:
        means : [m1, m2, ..., mK]  (mean for each OFF segment)
        vars  : [v1, v2, ..., vK]  (variance for each OFF segment)
    """
    assert len(signal) == len(status)
    N = len(signal)

    means = []
    vars_ = []

    in_off = False
    start = 0

    for i in range(N):
        if status[i] == 0:
            if not in_off:
                in_off = True
                start = i
        else:
            if in_off:
                end = i
                seg = signal[start:end]
                if len(seg) >= 2:
                    means.append(float(seg.mean()))
                    vars_.append(float(seg.var(ddof=0)))
                # nếu seg quá ngắn (<2) thì bỏ qua
                in_off = False

    # Last OFF segment 
    if in_off:
        end = N
        seg = signal[start:end]
        if len(seg) >= 2:
            means.append(float(seg.mean()))
            vars_.append(float(seg.var(ddof=0)))

    return np.array(means, dtype=np.float32), np.array(vars_, dtype=np.float32)


# =========================
# 3. Align 2 series 
# =========================
def align_series(a: np.ndarray, b: np.ndarray, mode: str = "truncate_min"):
    len_a, len_b = len(a), len(b)

    if len_a == len_b:
        return a, b

    if mode == "truncate_min":
        L = min(len_a, len_b)
        return a[:L], b[:L]

    # pad_last
    if len_a < len_b:
        pad_len = len_b - len_a
        pad_val = a[-1] if len_a > 0 else 0.0
        a_pad = np.concatenate([a, np.full(pad_len, pad_val, dtype=a.dtype)])
        return a_pad, b
    else:
        pad_len = len_a - len_b
        pad_val = b[-1] if len_b > 0 else 0.0
        b_pad = np.concatenate([b, np.full(pad_len, pad_val, dtype=b.dtype)])
        return a, b_pad

import numpy as np

def upsample_variance(vars_smooth: np.ndarray, target_len: int) -> np.ndarray:
    """
    Upsample 1D array vars_smooth to have the same target_len using linear interpolation.

    Parameters
    ----------
    vars_smooth : np.ndarray, shape (K,)
      
    target_len : int
      
    Returns
    -------
    upsampled : np.ndarray, shape (target_len,)
        
    """
    vars_smooth = np.asarray(vars_smooth, dtype=np.float32)
    K = len(vars_smooth)

    if K == 0:
        return np.zeros(target_len, dtype=np.float32)
    if K == 1:
        return np.full(target_len, vars_smooth[0], dtype=np.float32)

    x_old = np.arange(K, dtype=np.float32)
    x_new = np.linspace(0, K - 1, target_len, dtype=np.float32)

    upsampled = np.interp(x_new, x_old, vars_smooth).astype(np.float32)
    return upsampled



# =========================
# 4. MAIN
# =========================
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # ---- Load 2 run ----
    Ta0, Tw0, To0, Ts0, Tr0, status0 = load_run("../data_coldroom/data_train0")
    Ta1, Tw1, To1, Ts1, Tr1, status1 = load_run("../data_coldroom/data_train1")
    Ta_val, Tw_val, To_val, Ts_val, Tr_val, status_val = load_run("../data_coldroom/data_validation")
    


    SIGNAL_NAME = "Ta"
    sig0 = Ta0
    sig1 = Ta1

    means0, vars0 = compute_off_segment_stats(sig0, status0)
    means1, vars1 = compute_off_segment_stats(sig1, status1)
    means0 = means0[1:]
    means1 = means1[1:]
    vars0 = vars0[1:]
    vars1 = vars1[1:]

    vars0_val = compute_off_segment_stats(Ta_val, status_val)[1][1:]
    

    print("Run0 - số OFF segments:", len(means0))
    print("Run1 - số OFF segments:", len(means1))
    

    
    from scipy.signal import savgol_filter
    
    vars_smooth0 = savgol_filter(vars0, 51, 3)
    vars_smooth1 = savgol_filter(vars1, 51, 3)
    vars_smooth_val = savgol_filter(vars0_val, 51, 3)

    # Giả sử vars_smooth0, vars_smooth1 đã có
    min_len_Ta = min(len(Ta0), len(Ta1))
    min_len_var = min(len(vars_smooth0), len(vars_smooth1))
    #mean_profile = vars_smooth0
    #mean_profile = vars_smooth_val
    
    def build_var_profile(base, target_len):
        base = np.asarray(base, dtype=np.float32)
        Lb = len(base)
        if target_len <= Lb:
            return base[:target_len]
        pad_len = target_len - Lb
        pad_val = float(base[-1])
        pad = np.full(pad_len, pad_val, dtype=np.float32)
        return np.concatenate([base, pad], axis=0)


    N0   = len(Ta0)
    N1   = len(Ta1)
    Nval = len(Ta_val)	

    var_profile0         = upsample_variance(vars_smooth0, N0)
    var_profile1         = upsample_variance(vars_smooth1, N1)

    data_path0 = 'data_train0'
    data_path1 = 'data_train1'
    data_path_val = 'data_validation'
    np.save("./" + data_path0 + "/var_profile_Ta.npy", var_profile0)
    np.save("./" + data_path1 + "/var_profile_Ta.npy", var_profile1)

    mean_profile = 0.5 * (vars_smooth0[:min_len_var] + vars_smooth1[:min_len_var])

    base = upsample_variance(mean_profile, min_len_Ta)
    var_profile_validation = build_var_profile(base, Nval)

    np.save("./" + data_path_val + "/var_profile_Ta.npy", var_profile_validation)
    
    plt.plot(var_profile0)
    plt.plot(var_profile1)
    plt.plot(var_profile_validation)
    plt.show()



    # =========================
    # =========================
    seg_idx0 = np.arange(len(vars0))
    seg_idx1 = np.arange(len(vars1))

    plt.figure(figsize=(12, 6))

    # Run 0
    plt.subplot(2, 1, 1)
    plt.plot(seg_idx0, vars0, label="run0")
    plt.plot(seg_idx0, vars_smooth0, label="run0 smoothed", color='red')
    plt.xlabel("OFF segment index (run0)")
    plt.ylabel("Variance of " + SIGNAL_NAME)
    plt.title(f"Segment-wise variance (OFF) - data_train0")
    plt.grid(True, alpha=0.3)

    # Run 1
    plt.subplot(2, 1, 2)
    plt.plot(seg_idx1, vars1, label="run1", color='tab:orange')
    plt.xlabel("OFF segment index (run1)")
    plt.plot(seg_idx1, vars_smooth1, label="run1 smoothed", color='red')
    plt.ylabel("Variance of " + SIGNAL_NAME)
    plt.title(f"Segment-wise variance (OFF) - data_train1")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =========================
    # =========================
    seg_idx0 = np.arange(len(means0))
    seg_idx1 = np.arange(len(means1))

    plt.figure(figsize=(12, 6))

    # Run 0
    plt.subplot(2, 1, 1)
    plt.plot(seg_idx0, means0, label="run0")
    plt.xlabel("OFF segment index (run0)")
    plt.ylabel("Mean of " + SIGNAL_NAME)
    plt.title(f"Segment-wise mean (OFF) - data_train0")
    plt.grid(True, alpha=0.3)

    # Run 1
    plt.subplot(2, 1, 2)
    plt.plot(seg_idx1, means1, label="run1", color='tab:orange')
    plt.xlabel("OFF segment index (run1)")
    plt.ylabel("Mean of " + SIGNAL_NAME)
    plt.title(f"Segment-wise mean (OFF) - data_train1")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

