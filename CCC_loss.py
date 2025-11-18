import torch
import torch.nn as nn

class CCCLoss(nn.Module):
    """
    Computes the Concordance Correlation Coefficient (CCC) loss.
    The CCC measures the agreement between two variables, for example, to evaluate reproducibility.
    It is composed of a correlation term and a term that measures the deviation from the identity line.

    This implementation is designed for 4D tensors (e.g., weather data) with the shape:
    (batch_size, n_lat, n_lon, n_variables).

    The loss is calculated by computing the CCC for each variable's spatial map individually
    and then averaging the results. The final loss is 1 - mean_ccc.
    This version is vectorized to avoid a for-loop over variables for performance.
    """
    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon (float): A small value to add to the denominator for numerical stability.
        """
        super(CCCLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the CCC loss in a vectorized manner.

        Args:
            y_pred (torch.Tensor): The predicted tensor of shape (batch_size, n_lat, n_lon, n_variables).
            y_true (torch.Tensor): The ground truth tensor of shape (batch_size, n_lat, n_lon, n_variables).

        Returns:
            torch.Tensor: A single scalar value representing the average CCC loss across all variables and the batch.
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Input shapes must match. Got {y_pred.shape} and {y_true.shape}")
        
        if y_pred.dim() != 4:
            raise ValueError(f"Input tensors must be 4-dimensional. Got {y_pred.dim()} dimensions.")

        batch_size, n_lat, n_lon, n_variables = y_pred.shape
        
        # --- Reshape tensors to combine spatial dimensions and separate variables ---
        # Permute to bring variables to the second dimension: (B, V, H, W)
        pred_permuted = y_pred.permute(0, 3, 1, 2)
        true_permuted = y_true.permute(0, 3, 1, 2)

        # Flatten spatial dimensions: (B, V, H*W)
        pred_flat = pred_permuted.contiguous().view(batch_size, n_variables, -1)
        true_flat = true_permuted.contiguous().view(batch_size, n_variables, -1)

        # --- Calculate statistics for all variables at once ---
        # The 'dim=2' argument computes the statistic across the flattened spatial dimension
        # The resulting shape for all stats will be (batch_size, n_variables)
        mean_pred = torch.mean(pred_flat, dim=2)
        mean_true = torch.mean(true_flat, dim=2)

        # Use biased variance (unbiased=False) for consistency with covariance calculation
        var_pred = torch.var(pred_flat, dim=2, unbiased=False)
        var_true = torch.var(true_flat, dim=2, unbiased=False)

        # --- Calculate covariance ---
        # E[(X - E[X])(Y - E[Y])]
        # unsqueeze(2) adds a dimension for broadcasting: (B, V, 1)
        pred_centered = pred_flat - mean_pred.unsqueeze(2)
        true_centered = true_flat - mean_true.unsqueeze(2)
        covariance = torch.mean(pred_centered * true_centered, dim=2)

        # --- Calculate the Concordance Correlation Coefficient for all variables ---
        # Formula: (2 * cov) / (var_pred + var_true + (mean_pred - mean_true)^2)
        numerator = 2 * covariance
        denominator = var_pred + var_true + (mean_pred - mean_true)**2 + self.epsilon
        
        ccc_scores = numerator / denominator # Shape: (batch_size, n_variables)
        
        # The loss is 1 minus the CCC score
        loss = 1.0 - ccc_scores
        
        # --- Average across all variables and the batch to get a single scalar loss value ---
        final_loss = torch.mean(loss)
        
        return final_loss
