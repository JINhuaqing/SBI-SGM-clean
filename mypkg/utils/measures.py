import numpy as np


def lin_R_fn(x, y):
    """
    For both torch and np
    Calculate the linear correlation coefficient (Lin's R) between x and y.
    
    Args:
    x: torch.Tensor, shape (batch_size, num_features)/(num_feature)
    y: torch.Tensor, shape (batch_size, num_features)/(num_feature)
    
    Returns:
    ccc: torch.Tensor, shape (batch_size,)
    """
    assert x.shape == y.shape, "x and y should have the same shape"
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
    x_bar = x.mean(axis=-1, keepdims=True)
    y_bar = y.mean(axis=-1, keepdims=True)
    num = 2*((x-x_bar)*(y-y_bar)).sum(axis=-1);
    den = (x**2).sum(axis=-1) + (y**2).sum(axis=-1) - (2 * x.shape[-1] * x_bar * y_bar).squeeze()
    ccc = num/den;
    return ccc


def reg_R_fn(x, y):
    """Calculate pearons'r in batch, for both numpy and torch
    Args:
    x: torch.Tensor, shape (batch_size, num_features)/(num_feature)
    y: torch.Tensor, shape (batch_size, num_features)/(num_feature)
    Returns:
    corrs: torch.Tensor, shape (batch_size,)
    """
    assert x.shape == y.shape, "x and y should have the same shape"
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
    x_mean = x.mean(axis=-1, keepdims=True)
    y_mean = y.mean(axis=-1, keepdims=True)
    num = ((x- x_mean)*(y-y_mean)).sum(axis=-1)
    den = np.sqrt(((x- x_mean)**2).sum(axis=-1)*((y-y_mean)**2).sum(axis=-1))
    corrs = num/den
    return corrs

