import numpy as np


def implicit_als(y, factors, alpha, reg, iterations, seed):
    rng = np.random.default_rng(seed)
    n_users, n_items = y.shape
    q = rng.normal(0, 0.05, size=(n_items, factors))
    p = np.zeros((n_users, factors))
    eye = np.eye(factors)
    user_pos = [np.flatnonzero(y[u]) for u in range(n_users)]
    item_pos = [np.flatnonzero(y[:, i]) for i in range(n_items)]
    for _ in range(iterations):
        qtq = q.T @ q
        for u, pos in enumerate(user_pos):
            qp = q[pos]
            p[u] = np.linalg.solve(qtq + alpha * (qp.T @ qp) + reg * eye, (1 + alpha) * qp.sum(axis=0))
        ptp = p.T @ p
        for i, pos in enumerate(item_pos):
            if len(pos) == 0:
                q[i] = 0
                continue
            pp = p[pos]
            q[i] = np.linalg.solve(ptp + alpha * (pp.T @ pp) + reg * eye, (1 + alpha) * pp.sum(axis=0))
    return p @ q.T


def weighted_als(target, confidence, factors, reg, iterations, seed):
    rng = np.random.default_rng(seed)
    n_users, n_items = target.shape
    q = rng.normal(0, 0.05, size=(n_items, factors))
    p = np.zeros((n_users, factors))
    eye = np.eye(factors)
    for _ in range(iterations):
        for u in range(n_users):
            cq = confidence[u, :, None] * q
            p[u] = np.linalg.solve(q.T @ cq + reg * eye, q.T @ (confidence[u] * target[u]))
        for i in range(n_items):
            cp = confidence[:, i, None] * p
            q[i] = np.linalg.solve(p.T @ cp + reg * eye, p.T @ (confidence[:, i] * target[:, i]))
    return p @ q.T
