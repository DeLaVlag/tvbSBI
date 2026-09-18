import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

def do_sbi(theta, x, feature_names=None):

    # ---- Convert dataset ----
    thetas = torch.tensor(theta, dtype=torch.float32)
    xs = torch.tensor(x, dtype=torch.float32)

    x_mean = xs.mean(dim=0)
    x_std = xs.std(dim=0) + 1e-8

    xs = (xs - x_mean) / x_std

    print("[do_sbi] xs shape:", xs.shape)
    if feature_names is not None:
        dfa_indices = [i for i, name in enumerate(feature_names)
                       if name.startswith("dfa_curve_pca_")]
        if dfa_indices:
            dfa_curve = xs[:, dfa_indices]
            print("[do_sbi] dfa_curve block:",
                  dfa_curve.min(), dfa_curve.max(), dfa_curve.std())
            print("per-dim std:", dfa_curve.std(dim=0))

    # ---- Define prior (must match simulation prior) ----

    # for the LB:
    # slh = [.2, 1.2,  # coupling
    #        # -6, -3   ,  # weight_noise
    #        -6, -3,  # weight_noise
    #        .25, .55,  # rNMDA
    #        1.0, 20.0,  # globalspeed
    #        1.0, 1.3,  # tau_K Effect: change frequency & recovery dynamics.
    #        0.5, 0.9,  # phi Effect: change frequency & recovery dynamics.
    #        0.2, 2.5,  # dV
    #        0., 0.  # nothing
    #        ]

    prior = BoxUniform(
        low=torch.tensor([0.2, -6, .25, 1.0, 1.0, .5, .2]),   # example bounds
        high=torch.tensor([1.2, -3, .55, 20.0, 1.3, .9, 2.5])
    )

    # ---- Initialize inference ----
    inference = SNPE(prior=prior)

    # ---- Train density estimator ----
    density_estimator = inference.append_simulations(thetas, xs).train()

    # ---- Build posterior ----
    # for reals
    posterior = inference.build_posterior(density_estimator)

    # for testing
    # posterior = inference.build_posterior(
    #     density_estimator,
    #     sample_with="mcmc"
    # )

    return x_mean, x_std, xs, prior, posterior, density_estimator


######################################### Testing ###############################################

def posterior_width_test(posterior, xs, n_samples=500):
    widths = []

    for i in range(len(xs)):
        x_i = xs[i]
        samples = posterior.sample((n_samples,), x=x_i)

        std = samples.std(dim=0)  # per parameter
        widths.append(std.cpu())

    widths = torch.stack(widths)  # (N, n_params)
    return widths


######################################### Saving and Loading ###############################################

import torch
import pickle
from sbi.inference import SNPE
from sbi.utils import BoxUniform


# =========================
# SAVE FUNCTION
# =========================
def save_sbi_model(
    path,
    density_estimator,
    prior,
    x_mean,
    x_std,
    pca=None,
):
    """
    Save trained SBI model components.
    """

    save_dict = {
        "density_estimator_state": density_estimator.state_dict(),
        "prior_low": prior.low,
        "prior_high": prior.high,
        "x_mean": x_mean,
        "x_std": x_std,
        "theta_dim": prior.low.shape[0],
        "x_dim": x_mean.shape[0],
    }

    torch.save(save_dict, f"{path}/sbi_model.pt")

    # Save PCA separately (if used)
    if pca is not None:
        with open(f"{path}/pca.pkl", "wb") as f:
            pickle.dump(pca, f)


# =========================
# LOAD FUNCTION
# =========================
def load_sbi_model(path, device="cpu"):
    """
    Load SBI model and rebuild posterior.
    """

    checkpoint = torch.load(f"{path}/sbi_model.pt", map_location=device)

    # ---- Recreate prior ----
    prior = BoxUniform(
        low=checkpoint["prior_low"].to(device),
        high=checkpoint["prior_high"].to(device),
    )

    # ---- Rebuild inference ----
    inference = SNPE(prior=prior)

    # ---- Recreate dummy inputs for network ----
    theta_dim = checkpoint["theta_dim"]
    x_dim = checkpoint["x_dim"]

    # ---- Use REAL data for initialization ----
    dummy_theta = checkpoint["prior_low"].unsqueeze(0).to(device)

    # CRITICAL: add small noise to avoid zero std
    dummy_theta = dummy_theta + 1e-6 * torch.randn_like(dummy_theta)

    dummy_x = checkpoint["x_mean"].unsqueeze(0).to(device)
    dummy_x = dummy_x + 1e-6 * torch.randn_like(dummy_x)

    density_estimator = inference._build_neural_net(
        dummy_theta,
        dummy_x
    )

    # ---- Load weights ----
    density_estimator.load_state_dict(
        checkpoint["density_estimator_state"]
    )

    density_estimator.to(device)

    # ---- Rebuild posterior ----
    posterior = inference.build_posterior(density_estimator)

    # ---- Load normalization ----
    x_mean = checkpoint["x_mean"].to(device)
    x_std = checkpoint["x_std"].to(device)

    # ---- Load PCA if exists ----
    try:
        with open(f"{path}/pca.pkl", "rb") as f:
            pca = pickle.load(f)
    except FileNotFoundError:
        pca = None

    return {
        "posterior": posterior,
        "prior": prior,
        "x_mean": x_mean,
        "x_std": x_std,
        "pca": pca
    }


# =========================
# FEATURE PREP FUNCTION
# =========================
def prepare_x(x_raw, x_mean, x_std, pca=None):
    """
    Apply PCA (if any) and normalize features.
    """

    if pca is not None:
        x_raw = pca.transform(x_raw)

    x = torch.tensor(x_raw, dtype=torch.float32)

    x = (x - x_mean) / (x_std + 1e-8)

    return x


# =========================
# INFERENCE HELPER
# =========================
def infer_theta(posterior, x, n_samples=1000):
    """
    Sample parameters from posterior.
    """
    return posterior.sample((n_samples,), x=x)

