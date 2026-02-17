import h5py
import numpy as np
import torch


class Camera:
    def __init__(self, K, R, t):
        self.K = K
        self.R = R
        self.t = t

    @classmethod
    def from_calibration_file(cls, path: str):
        with h5py.File(path, "r") as f:
            K = torch.tensor(np.array(f["K"]), dtype=torch.float32)
            R = torch.tensor(np.array(f["R"]), dtype=torch.float32)
            T = torch.tensor(np.array(f["T"]), dtype=torch.float32)

        return cls(K, R, T)

    @property
    def K_inv(self):
        return self.K.inverse()

    def to_cameradict(self):
        fx = self.K[0, 0].item()
        fy = self.K[1, 1].item()
        cx = self.K[0, 2].item()
        cy = self.K[1, 2].item()

        params = {
            "model": "PINHOLE",
            "width": int(cx * 2),
            "height": int(cy * 2),
            "params": [fx, fy, cx, cy],
        }

        return params

    def __repr__(self):
        return f"ImageData(K={self.K}, R={self.R}, t={self.t})"


def cameras2F(cam1: Camera, cam2: Camera) -> torch.Tensor:
    E = cameras2E(cam1, cam2)
    return cam2.K_inv.T @ E @ cam1.K_inv


def cameras2E(cam1: Camera, cam2: Camera) -> torch.Tensor:
    R = cam2.R @ cam1.R.T
    T = cam2.t - R @ cam1.t
    return cross_product_matrix(T) @ R


def cross_product_matrix(v) -> torch.Tensor:
    """Following en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication."""

    return torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=v.dtype,
        device=v.device,
    )
