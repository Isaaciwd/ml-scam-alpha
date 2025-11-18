import torch
import torch.nn as nn

class BarlowTwinsLoss(nn.Module):
    """
    Implementation of the Barlow Twins Loss Function.
    This loss function is designed to work with self-supervised learning setups.
    It encourages the embeddings of augmented versions of a sample to be similar,
    while minimizing the redundancy between the components of these embeddings.

    Args:
        lambda_ (float): The weight for the redundancy reduction term.
                         A value of 5e-3 is often used in literature.
        eps (float): A small epsilon value for numerical stability in normalization.
    """
    def __init__(self, lambda_=5e-3, eps=1e-8):
        super().__init__()
        self.lambda_ = lambda_
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Barlow Twins loss between two batches of embeddings.

        Args:
            z_a (torch.Tensor): The first batch of embeddings, with shape
                                [batch_size, embedding_dim].
            z_b (torch.Tensor): The second batch of embeddings, with shape
                                [batch_size, embedding_dim].

        Returns:
            torch.Tensor: The calculated Barlow Twins loss.
        """
        # Ensure the inputs are of the same shape
        assert z_a.shape == z_b.shape, "Input tensors z_a and z_b must have the same shape."

        orig_dtype = z_a.dtype

        with torch.cuda.amp.autocast(enabled=False):
            z_a = torch.nan_to_num(z_a.to(dtype=torch.float32))
            z_b = torch.nan_to_num(z_b.to(dtype=torch.float32))

            batch_size, embedding_dim = z_a.shape

            mean_a = z_a.mean(dim=0)
            mean_b = z_b.mean(dim=0)
            std_a = z_a.std(dim=0, unbiased=False)
            std_b = z_b.std(dim=0, unbiased=False)

            denom_a = std_a.clamp_min(self.eps)
            denom_b = std_b.clamp_min(self.eps)

            # 1. Normalize the embeddings along the batch dimension
            z_a_norm = torch.nan_to_num((z_a - mean_a) / denom_a)
            z_b_norm = torch.nan_to_num((z_b - mean_b) / denom_b)

            # 2. Calculate the cross-correlation matrix
            # The matrix C will have shape [embedding_dim, embedding_dim]
            c = (z_a_norm.T @ z_b_norm) / max(batch_size, 1)

            # 3. Calculate the two components of the loss
            on_diag = torch.diagonal(c)
            invariance_loss = ((on_diag - 1) ** 2).mean()

            off_diag = c - torch.diag_embed(on_diag)
            redundancy_loss = (off_diag ** 2).sum() / max(embedding_dim * (embedding_dim - 1), 1)

            # 4. Combine the terms with the lambda weighting factor
            total_loss = invariance_loss + self.lambda_ * redundancy_loss
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return total_loss.to(orig_dtype)