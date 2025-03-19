import torch
from typing import Union

def batch_uniform_noise(
    arr: torch.Tensor,
    indices: torch.Tensor,
    lower_bound: float = 0.02,
    upper_bound: Union[None, float] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Add noise to the input at indices as sampled uniformly random from [-lower_bound, lower_bound].
    if upper_bound is None, and [lower_bound, upper_bound] otherwise. 2D array [batch_size, n_features]

    Parameters
    ----------
    arr: torch.Tensor
         Tensor to be perturbed.
    indices: torch.Tensor
        Tensor-like, with a subset shape of arr.
    lower_bound: float
            The lower bound for uniform sampling.
    upper_bound: float, optional
            The upper bound for uniform sampling.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    arr_perturbed: torch.Tensor
         The tensor which some of its indices have been perturbed.
    """

    batch_size = arr.shape[0]
    arr_perturbed = arr.clone() # TODO: Check if this is necessary

    # Sample the noise.
    if upper_bound is None:
        noise = torch.empty_like(arr).uniform_(-lower_bound, lower_bound)
    else:
        assert (
            upper_bound > lower_bound
        ), "Parameter 'upper_bound' needs to be larger than 'lower_bound', " "but {} <= {}".format(
            upper_bound, lower_bound
        )
        noise = torch.empty_like(arr).uniform_(lower_bound, upper_bound)

    # Perturb the array.
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, indices.size(1))
    arr_perturbed[batch_indices, indices] = (arr_perturbed + noise)[batch_indices, indices]

    return arr_perturbed

def difference(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Calculate the difference between two images (or explanations).

    Parameters
    ----------
    a: torch.Tensor
         The first tensor to use for similarity scoring.
    b: torch.Tensor
         The second tensor to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    torch.Tensor
        The difference in each element.
    """
    return a - b