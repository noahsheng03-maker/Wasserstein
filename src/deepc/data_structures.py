from __future__ import annotations

import math

import numpy as np


def page_matrix(sequence: np.ndarray, depth: int) -> np.ndarray:
    """Return the Page matrix with block depth `depth`.

    Parameters
    ----------
    sequence:
        Array of shape (signal_dim, horizon).
    depth:
        Block depth L.
    """
    signal_dim, horizon = sequence.shape
    n_cols = horizon // depth
    if n_cols <= 0:
        raise ValueError("Page matrix requires horizon >= depth.")

    trimmed = sequence[:, : n_cols * depth]
    blocks = [
        trimmed[:, col * depth : (col + 1) * depth].reshape(signal_dim * depth, order="F")
        for col in range(n_cols)
    ]
    return np.stack(blocks, axis=1)


def hankel_matrix(sequence: np.ndarray, depth: int) -> np.ndarray:
    """Return the block Hankel matrix with depth `depth`."""
    signal_dim, horizon = sequence.shape
    n_cols = horizon - depth + 1
    if n_cols <= 0:
        raise ValueError("Hankel matrix requires horizon >= depth.")

    blocks = [
        sequence[:, col : col + depth].reshape(signal_dim * depth, order="F")
        for col in range(n_cols)
    ]
    return np.stack(blocks, axis=1)


def partition_data_matrix(
    u_sequence: np.ndarray,
    y_sequence: np.ndarray,
    t_ini: int,
    t_f: int,
    matrix_type: str = "page",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build and partition DeePC data matrices."""
    depth = t_ini + t_f
    matrix_type = matrix_type.lower()
    if matrix_type == "page":
        u_mat = page_matrix(u_sequence, depth)
        y_mat = page_matrix(y_sequence, depth)
    elif matrix_type == "hankel":
        u_mat = hankel_matrix(u_sequence, depth)
        y_mat = hankel_matrix(y_sequence, depth)
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")

    signal_u = u_sequence.shape[0]
    signal_y = y_sequence.shape[0]
    u_p = u_mat[: signal_u * t_ini, :]
    u_f = u_mat[signal_u * t_ini :, :]
    y_p = y_mat[: signal_y * t_ini, :]
    y_f = y_mat[signal_y * t_ini :, :]
    return u_p, u_f, y_p, y_f


def candidate_columns_for_horizon(total_points: int, t_ini: int, t_f: int, matrix_type: str) -> int:
    depth = t_ini + t_f
    if matrix_type.lower() == "page":
        return total_points // depth
    if matrix_type.lower() == "hankel":
        return max(0, total_points - depth + 1)
    raise ValueError(f"Unsupported matrix type: {matrix_type}")
