import numpy as np
from scipy import optimize, sparse


def db_exp_decay(
    x_array,
    const,
    amp_slow,
    tau_slow,
    amp_fast,
    tau_fast,
):
    """Double exponential decay equation with constant offset. Tau_slow is just
    tau_fast*tau_multiplier

    Args:
        x_array (_type_): _description_
        const (_type_): _description_
        amp_fast (_type_): _description_
        tau_fast (_type_): _description_
        amp_slow (_type_): _description_
        tau_slow (_type_): _description_

    Returns:
        _type_: _description_
    """
    y = (
        const
        + (amp_slow * np.exp((-x_array) / tau_slow))
        + (amp_fast * np.exp((-x_array) / tau_fast))
    )
    return y


def double_exp(signal, timestamp):
    max_sig = np.max(signal)
    # const, amp_slow, tau_slow, amp_fast, tau_fast
    inital_params = [max_sig / 2, max_sig / 4, 3600, max_sig / 4, 360]
    bounds = ([0, 0, 600, 0, 0], [max_sig, max_sig, 36000, max_sig, 3600])
    fit_params, parm_cov = optimize.curve_fit(
        db_exp_decay,
        timestamp,
        signal,
        p0=inital_params,
        bounds=bounds,
        maxfev=1000,
    )
    fit = db_exp_decay(timestamp, *fit_params)
    return fit


def WhittakerSmooth(x, w, lambda_, differences=1):
    """
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    """
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = sparse.eye(m, format="csc")
    D = (
        E[1:] - E[:-1]
    )  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = sparse.diags(w, 0, shape=(m, m))
    A = sparse.csc_matrix(W + (lambda_ * D.T * D))
    B = sparse.csc_matrix(W * X.T)
    background = sparse.linalg.spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    """
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    """
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * (abs(x)).sum() or i == itermax:
            if i == itermax:
                print("WARING max iteration reached!")
            break
        w[
            d >= 0
        ] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z


# Alternative arPLS from https://stackoverflow.com/questions/29156532/python-baseline-correction-library.
def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio:
        z = sparse.linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))

        crit = np.linalg.norm(w_new - w) / np.linalg.norm(w)

        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            print("Maximum number of iterations exceeded")
            break

    if full_output:
        info = {"num_iter": count, "stop_criterion": crit}
        return z, d, info
    else:
        return z


# %%
import matplotlib.pyplot as plt
import statsmodels.api as sm
from acquisition.photometry import ExpManager

# %%
pc_path = r"D:/Photometry/2022_07_07_photometry/1039085_N_Session5_PhotoData.csv"
path = (
    r"/Volumes/Backup/Photometry/2022_07_07_photometry/1039085_N_Session5_PhotoData.csv"
)

# %%
exp = ExpManager()
exp.load_data(pc_path)

# %%
y = exp.get_raw_data("1039085_N_Session5_PhotoData", "RawF_560_F1").to_numpy()

# %%
baseline = airPLS(y, lambda_=200000)
plt.plot(y)
plt.plot(baseline)

# %%
baseline = baseline_arPLS(y, ratio=1e-15)
plt.plot(y)
plt.plot(baseline)

# %%
plt.plot(y - baseline)
# %%
