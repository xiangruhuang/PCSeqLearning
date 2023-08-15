import torch
import numpy as np

def cartesian2spherical_np(xyz):
    """Cartesian coordinate system to Sphereical coordinate system
    Args:
        xyz [N, 3]
    Returns:
        r: radius of each point
        polar: angle between z-axis and x-y plane or inclination in range [0, pi]
        azimuth: rotation angle in x-y plane in range [-pi, pi]
    """
    r = np.linalg.norm(xyz, ord=2, axis=-1).astype(np.float32) # [N]
    r[r < 1e-4] = 1e-4
    polar = np.arccos(xyz[:, 2]/r).astype(np.float32)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0]).astype(np.float32)

    return r, polar, azimuth

def cartesian2spherical(xyz):
    """Cartesian coordinate system to Sphereical coordinate system
    Args:
        xyz [N, 3]
    Returns:
        r: radius of each point
        polar: angle between z-axis and x-y plane or inclination in range [0, pi]
        azimuth: rotation angle in x-y plane in range [-pi, pi]
    """
    r = xyz.norm(p=2, dim=-1)
    r = r.clamp(min=1e-4)
    polar = torch.acos(xyz[:, 2]/r)
    azimuth = torch.atan2(xyz[:, 1], xyz[:, 0])

    return r, polar, azimuth

def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [N, 3] / [N, G, 3]
    :return: (rho, theta, phi) [N, 3] / [N, G, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]  TODO Use range of [-1, 1] instead
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out

def xyz2sphere_np(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate
    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    :param xyz: [N, 3] / [N, G, 3]
    :return: (rho, theta, phi) [N, 3] / [N, G, 3]
    """
    rho = np.sqrt(np.sum(np.power(xyz, 2), axis=-1))[..., np.newaxis]
    rho = np.clip(rho, 0, None)  # range: [0, inf]
    theta = np.arccos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = np.arctan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]  TODO Use range of [-1, 1] instead
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = np.concatenate([rho, theta, phi], axis=-1).astype(np.float32)
    return out

def xyz2sphere_aug(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate (Augmentation by YZ & XZ direction)

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, theta, phi) [B, N, 3] / [B, N, G, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, 0, 1)  # range: [0, 1]

    # XY direction
    theta_xy = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi_xy = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # XZ direction
    theta_xz = torch.acos(xyz[..., 1, None] / rho)  # range: [0, pi]
    phi_xz = torch.atan2(xyz[..., 2, None], xyz[..., 0, None])  # range: [-pi, pi]
    # YZ direction
    theta_yz = torch.acos(xyz[..., 0, None] / rho)  # range: [0, pi]
    phi_yz = torch.atan2(xyz[..., 2, None], xyz[..., 1, None])  # range: [-pi, pi]

    # check nan
    idx = rho == 0
    theta_xy[idx] = 0
    theta_xz[idx] = 0
    theta_yz[idx] = 0

    theta = torch.cat([theta_xy, theta_xz, theta_yz], dim=-1)
    phi = torch.cat([phi_xy, phi_xz, phi_yz], dim=-1)

    if normalize:
        theta = theta / np.pi
        phi = phi / (2 * np.pi) + .5
    out = torch.cat([rho, theta, phi], dim=-1)
    return out


def normal2sphere(xyz, normalize=False):
    """
    Convert Normal Vector to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (theta, phi) [B, N, 2] / [B, N, G, 2]
    """
    theta = torch.acos(xyz[..., 2, None])  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]

    if normalize:
        theta = theta / np.pi
        phi = phi / (2 * np.pi) + .5
    out = torch.cat([theta, phi], dim=-1)
    return out


def sphere2normal(sphere):
    x = torch.cos(sphere[..., 1, None]) * torch.sin(sphere[..., 0, None])
    y = torch.sin(sphere[..., 1, None]) * torch.sin(sphere[..., 0, None])
    z = torch.cos(sphere[..., 0, None])
    return torch.cat([x, y, z], dim=-1)


def xyz2cylind(xyz, normalize=True):
    """
    Convert XYZ to Cylindrical Coordinate

    reference: https://en.wikipedia.org/wiki/Cylindrical_coordinate_system

    :param normalize: Normalize phi & z
    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, phi, z) [B, N, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz[..., :2], 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, 0, 1)  # range: [0, 1]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    z = xyz[..., 2, None]
    z = torch.clamp(z, -1, 1)  # range: [-1, 1]

    if normalize:
        phi = phi / (2 * np.pi) + .5
        z = (z + 1.) / 2.
    out = torch.cat([rho, phi, z], dim=-1)
    return out
