import torch

def hash_int(coords, dims):
    """
    Args:
        coords [V, dim]: non-negative integer coordinates
        dims [dim]: specified size of each dimension

    Returns:
        unique_coords [U, dim]: unique coordinates
        group_indices [V]: from element to group
    """
    assert isinstance(coords, torch.Tensor)
    assert (coords.dtype == torch.int32) or (coords.dtype == torch.int64)
    assert (dims.dtype == torch.int32) or (dims.dtype == torch.int64)
    
    num_dim = coords.shape[-1]
    assert num_dim == dims.shape[0]

    coord1d = coords.new_zeros(coords.shape[0])
    # from num_dim-D to 1D
    for i in range(num_dim):
        coord1d = coord1d * dims[i] + coords[:, i]
    
    unique_coord1d, inverse = torch.unique(coord1d, return_inverse=True)
    num_unique_coords = unique_coord1d.shape[0]
    
    unique_coords = coords.new_zeros(num_unique_coords, num_dim)
    # from 1D back to num_dim-D
    for i in range(num_dim-1, -1, -1):
        unique_coords[:, i] = unique_coord1d % dims[i]
        unique_coord1d = torch.div(unique_coord1d, dims[i], rounding_mode='floor')
    
    return unique_coords, inverse
