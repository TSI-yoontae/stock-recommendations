import numpy as np

from models.wmf import weighted_als


def mvecf_wmf(y, w, mu, sigma, args, seed):
    base_conf = args.negative_confidence + (args.positive_confidence - args.negative_confidence) * y
    diag = np.diag(sigma)
    cov_excl = w @ sigma - w * diag[None, :]
    c_mv = 0.5 * args.gamma_mv * args.lambda_mv * diag[None, :]
    c_tilde = base_conf + c_mv
    y_tilde = (
        base_conf * y
        + 0.5 * args.lambda_mv * mu[None, :]
        - 0.25 * args.gamma_mv * args.lambda_mv * cov_excl
    ) / c_tilde
    scores = weighted_als(
        y_tilde,
        c_tilde,
        args.factors,
        args.reg,
        args.iterations,
        seed + 10999,
    )
    return scores, y_tilde, c_tilde
