import numpy as np
from . import primitive_fitting_utils
import time

from scipy.optimize import minimize, least_squares
from typing import Optional, Dict


def fit_plane_numpy(points : np.ndarray):
    start = time.time()

    c = points.mean(axis = 0)
    X = points - c

    try:
        U, S, Vh = np.linalg.svd(X, full_matrices = False)
    except np.linalg.LinAlgError:
        end = time.time()
        return {
            "surface_type": "plane",
            "error": float("inf"),
            "params": {
                "a": np.array([1.0, 0.0, 0.0]),
                "d": 0.0,
            },
            "metadata": {
                "fitting_time_seconds": end - start,
                "svd_converged": False,
            },
        }
    # The plane normal is the eigenvector of the smallest eigenvalue
    a = Vh[-1]
    d = a @ c

    end = time.time()

    res = {
        "surface_type": "plane",
        "error": primitive_fitting_utils.plane_error(points, a, d),
        "params": {
            "a": a,
            "d": d
        },
        "metadata": {
            "fitting_time_seconds": end - start
        }
    }

    return res

def fit_sphere_numpy(points: np.ndarray, rcond : float = 1e-6):
    # rcond: singular values below rcond * largest are treated as zero
    start = time.time()
    A = np.concatenate((2 * points, np.ones((points.shape[0], 1))), axis = 1)
    y = (points ** 2).sum(axis = 1)
    try:
        w, _, rank, _ = np.linalg.lstsq(A, y, rcond = rcond)
    except np.linalg.LinAlgError:
        end = time.time()
        return {
            "surface_type": "sphere",
            "error": float("inf"),
            "params": {
                "center": np.zeros(3),
                "radius": 0.0,
            },
            "metadata": {
                "fitting_time_seconds": end - start,
                "lstsq_converged": False,
            },
        }

    center = w[:3]
    radius = w[3] + (center ** 2).sum()
    radius = radius ** 0.5

    end = time.time()
    res = {
        "surface_type": "sphere",
        "error": primitive_fitting_utils.sphere_error(points, center, radius),
        "params": {
            "center": center,
            "radius": radius
        },
        "metadata": {
            "fitting_time_seconds": end - start
        }
    }

    return res

def fit_cylinder(data, guess_angles = None):
    def direction(theta, phi):
        return np.array(
            [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
        )

    def projection_matrix(w):
        return np.identity(3) - np.dot(np.reshape(w, (3, 1)), np.reshape(w, (1, 3)))

    def skew_matrix(w):
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    def calc_A(Ys):
        return sum(np.dot(np.reshape(Y, (3, 1)), np.reshape(Y, (1, 3))) for Y in Ys)

    def calc_A_hat(A, S):
        return np.dot(S, np.dot(A, np.transpose(S)))

    def preprocess_data(Xs_raw):
        n = len(Xs_raw)
        Xs_raw_mean = sum(X for X in Xs_raw) / n

        return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean

    def G(w, Xs):
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        u = sum(np.dot(Y, Y) for Y in Ys) / n
        v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

        return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)

    def C(w, Xs):
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

    def r(w, Xs):
        n = len(Xs)
        P = projection_matrix(w)
        c = C(w, Xs)

        return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

    start = time.time()
    Xs, t = preprocess_data(data)

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    best_fit = None
    best_score = float("inf")

    for sp in start_points:
        fitted = minimize(
            lambda x: G(direction(x[0], x[1]), Xs), sp, method = "Powell", tol = 1e-6
        )

        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    # Powell returns NaN on degenerate clusters; NaN < inf is False, so best_fit stays None and fitness is reported as inf
    if best_fit is None:
        end = time.time()
        return (
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            0.0,
            float("inf"),
            end - start,
        )

    w = direction(best_fit.x[0], best_fit.x[1])
    end = time.time()

    return w, C(w, Xs) + t, r(w, Xs), best_fit.fun, end - start

def fit_cylinder_optimized(data, guess_angles = None):
    def direction(theta, phi):
        return np.array([
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta)
        ])

    def preprocess_data(Xs_raw):
        X = np.array(Xs_raw)  # n × 3 matrix
        X_mean = X.mean(axis = 0)
        X_centered = X - X_mean
        return X_centered, X_mean

    def G_vectorized(w, X):
        n = X.shape[0]

        P = np.eye(3) - np.outer(w, w) # [3, 3]

        Y = X @ P.T # [N, 3]

        Y_norm_sq = np.sum(Y * Y, axis = 1)  # [N]

        A = Y.T @ Y # [3, 3]

        S = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])

        A_hat = S @ A @ S.T # [3, 3]

        trace_AA_hat = np.trace(A_hat @ A)
        u = Y_norm_sq.mean()

        weighted_Y = X.T @ Y_norm_sq  # [3,]
        v = A_hat @ weighted_Y / trace_AA_hat # [3,]

        residuals = Y_norm_sq - u - 2 * (X @ v)

        # The original does not scale residuals by 1/n; the morphology of the direction scoring function is unchanged
        return np.sum(residuals ** 2) / n

    def C_vectorized(w, X):
        n = X.shape[0]
        P = np.eye(3) - np.outer(w, w)
        Y = X @ P.T
        Y_norm_sq = np.sum(Y * Y, axis = 1)

        A = Y.T @ Y
        S = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        A_hat = S @ A @ S.T

        weighted_Y = X.T @ Y_norm_sq
        return A_hat @ weighted_Y / np.trace(A_hat @ A)

    def r_vectorized(w, X):
        P = np.eye(3) - np.outer(w, w)
        c = C_vectorized(w, X)
        d = X - c
        perp_dist_sq = np.sum(d @ P * d, axis = 1).mean()
        return perp_dist_sq ** 0.5

    start = time.time()
    X, t = preprocess_data(data)

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    best_fit = None
    best_score = float("inf")
    for sp in start_points:
        fitted = minimize(
            lambda angles: G_vectorized(direction(angles[0], angles[1]), X),
            sp,
            method = "Powell",
            tol = 1e-6
        )
        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    # Powell returns NaN on degenerate clusters; NaN < inf is False, so best_fit stays None and the error is reported as inf
    if best_fit is None:
        end = time.time()
        return {
            "surface_type": "cylinder",
            "error": float("inf"),
            "params": {
                "a": np.array([1.0, 0.0, 0.0]),
                "center": np.zeros(3),
                "radius": 0.0,
            },
            "metadata": {
                "best_fit": float("inf"),
                "optimizer_iterations": 0,
                "fitting_time_seconds": end - start,
                "optimizer_converged": False,
            },
        }

    w = direction(best_fit.x[0], best_fit.x[1])
    center = C_vectorized(w, X) + t
    radius = r_vectorized(w, X)
    end = time.time()

    res = {
        "surface_type": "cylinder",
        "error": primitive_fitting_utils.cylinder_error(data, center, w, radius),
        "params": {
            "a": w,
            "center": center,
            "radius": radius
        },
        "metadata": {
            "best_fit": best_fit.fun, # Not actually least squares fitness, just reparametrized axis component.
            "optimizer_iterations": best_fit.nit,
            "fitting_time_seconds": end - start,
            "optimizer_converged": best_fit.success
        }
    }

    return res

def cone_residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    theta = params[0]
    axis = params[1:4]
    vertex = params[4:7]

    axis_unit = axis / np.linalg.norm(axis)

    d = points - vertex
    M = np.cos(theta)**2 * np.eye(3) - np.outer(axis_unit, axis_unit)

    residuals = np.sum((d @ M) * d, axis = 1)

    return residuals

def cone_jacobian(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    theta = params[0]
    axis = params[1:4]
    vertex = params[4:7]

    axis_norm = np.linalg.norm(axis)
    axis_unit = axis / axis_norm

    n = len(points)
    J = np.zeros((n, 7))

    d = points - vertex
    M = np.cos(theta)**2 * np.eye(3) - np.outer(axis_unit, axis_unit)

    d_norm_sq = np.sum(d * d, axis=1)
    J[:, 0] = -np.sin(2 * theta) * d_norm_sq

    axis_dot_d = d @ axis_unit
    d_perp = d - axis_dot_d[:, np.newaxis] * axis_unit
    J[:, 1:4] = (-2 * axis_dot_d[:, np.newaxis] * d_perp) / axis_norm

    M_d = d @ M.T
    J[:, 4:7] = -2 * M_d

    return J

def fit_cone(points: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Dict:
    start = time.time()

    residual_fn = lambda p: cone_residuals(p, points)
    jacobian_fn = lambda p: cone_jacobian(p, points)

    if initial_guess is not None:
        initial_guesses = [initial_guess]

    else:
        centroid = points.mean(axis = 0)

        initial_guesses = [
            np.array([0.1, 1.0, 0.0, 0.0, *centroid]),
            np.array([0.1, 0.0, 1.0, 0.0, *centroid]),
            np.array([0.1, 0.0, 0.0, 1.0, *centroid])
        ]

    best_result = None
    best_cost = np.inf
    lower_bounds = [0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    upper_bounds = [np.pi / 2, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    for x0 in initial_guesses:
        try:
            result = least_squares(
                residual_fn,
                x0,
                jac = jacobian_fn,
                method = "trf",
                bounds = (lower_bounds, upper_bounds),
                ftol = 1e-10,
            )

            if result.success and result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except:
            continue

    end = time.time()

    if best_result is None:
        return {
        "surface_type": "cone",
        "error": float("inf"),
        "params": {
            "a": np.zeros(3),
            "v": np.zeros(3),
            "theta": 0
        },
        "metadata": {
            "fitting_time_seconds": end - start,
            "best_fit": float("inf"),
            "optimizer_converged": False,
            "optimizer_iterations": float("inf")
            }
        }   

    theta_opt = best_result.x[0]
    axis_opt = best_result.x[1:4]
    vertex_opt = best_result.x[4:7]

    axis_opt = axis_opt / np.linalg.norm(axis_opt)

    res = {
        "surface_type": "cone",
        "error": primitive_fitting_utils.cone_error(points, vertex_opt, axis_opt, theta_opt),
        "params": {
            "a": axis_opt,
            "v": vertex_opt,
            "theta": theta_opt
        },
        "metadata": {
            "fitting_time_seconds": end - start,
            "best_fit": best_result.cost,
            "optimizer_converged": best_result.success,
            "optimizer_iterations": best_result.nfev
        }
    }

    return res
