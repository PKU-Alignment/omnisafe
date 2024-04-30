import numpy as np
import torch


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array.

    Args:
    x (torch.Tensor): A torch tensor to be converted.

    Returns:
    np.ndarray: A numpy array representation of the input tensor.
    """
    return x.cpu().detach().double().numpy()


def to_tensor(
    x: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert a numpy array to a torch tensor of specified type and device.

    Args:
    x (np.ndarray): A numpy array to be converted.
    dtype (torch.dtype): The desired data type for the tensor.
    device (torch.device): The device to store the tensor on.
    requires_grad (bool): If True, gradients will be computed for operations involving this tensor.

    Returns:
    torch.Tensor: A torch tensor representation of the input array.
    """
    if type(x).__module__ != 'numpy':
        return x
    return torch.from_numpy(x).type(dtype).to(device).requires_grad_(requires_grad)


def sort_vertices_cclockwise(vertices: np.ndarray) -> np.ndarray:
    """Sort vertices of a 2D convex polygon in counter-clockwise direction.

    Args:
    vertices (np.ndarray): An array of shape (n_v, 2) where n_v is the number of vertices.

    Returns:
    np.ndarray: An array of vertices sorted in counter-clockwise direction.
    """
    assert vertices.shape[1] == 2, f'Vertices must each have dimension 2, got {vertices.shape[1]}'

    # Sort vertices
    polygon_center = vertices.sum(axis=0, keepdims=True) / vertices.shape[0]  # (1, d)
    rel_vecs = vertices - polygon_center
    thetas = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
    idxs = np.argsort(thetas)
    return vertices[idxs, :]
