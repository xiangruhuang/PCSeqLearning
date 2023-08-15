import numpy as np
import torch

def cross_op_np(r):
  """
  Return the cross operator as a matrix
  i.e. for input vector r \in \R^3
  output rX s.t. rX.dot(v) = np.cross(r, v)
  where rX \in \R^{3 X 3}
  """
  rX = np.zeros((3, 3))
  rX[0, 1] = -r[2]
  rX[0, 2] = r[1]
  rX[1, 2] = -r[0]
  rX = rX - rX.T
  return rX

def cross_op(r):
    """
    Args:
        r [N, 3]: vectors
    
    Returns:
        R [N, 3, 3]: cross operator matrices
    """
    r0, r1, r2 = r.T
    ones = torch.ones_like(r1)
    R = torch.stack([ones, -r2, r1,
                     r2, ones, -r0,
                     -r1, r0, ones], dim=-1).reshape(-1, 3, 3)
    return R

def rodrigues(r, eps=1e-6):
    """
    Args:
        r [N, 3]: vectors
    
    Returns:
        R [N, 3, 3]: cross operator matrices
    """
    Id = torch.diag_embed(torch.ones_like(r))
    theta = r.norm(p=2, dim=-1) 
    mask = theta > eps
    k = r[mask] / theta[mask, None] # [N_valid, 3]
    """ Rodrigues """

    R = Id.clone()
    kouter = k[:, :, None] * k[:, None, :] # [N_valid, 3, 3]
    cost = theta[mask].cos()[:, None, None] # [N_valid]
    sint = theta[mask].sin()[:, None, None] # [N_valid]
    R[mask] = cost*Id[mask] + sint*cross_op(k) + (1-cost)*kouter

    return R

def rodrigues_np(r):
  """
  Return the rotation matrix R as a function of (axis, angle)
  following Rodrigues rotation theorem.
  (axis, angle) are represented by an input vector r, where
  axis = r / l2_norm(r) and angle = l2_norm(r)
  """
  theta = np.linalg.norm(r, 2)
  if theta < 1e-12:
    return np.eye(3)
  k = r / theta
  """ Rodrigues """
  R = np.cos(theta)*np.eye(3) + np.sin(theta)*cross_op(k) + (1-np.cos(theta))*np.outer(k, k)
  return R
        
