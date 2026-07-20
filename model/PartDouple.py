import torch
import torch.nn as nn
import torch.nn.functional as F

class JointImportanceDetermination(nn.Module):
    """
    Joint Importance Determination (JID) Module with calibration loss.
    Learns per-joint importance and aligns with ground-truth distribution via MSE.
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_joints: int):
        """
        Args:
            in_channels (int): Dimension C_pre of pre-encoded joint features.
            hidden_dim (int): Hidden size for the internal FC layer.
            num_joints (int): Number of joints V.
        """
        super(JointImportanceDetermination, self).__init__()
        # Importance MLP
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        # Calibration loss
        self.mse_loss = nn.MSELoss()
        self.num_joints = num_joints

    def forward(self,
                joint_feats: torch.Tensor,
                k_gt: torch.Tensor = None
            ) -> torch.Tensor:
        # MLP layers
        x = self.fc1(joint_feats)             # B, T, V, D
        x = self.relu(x)                      # B, T, V, D
        logits = self.fc2(x)                  # B, T, V, 1
        weights = F.softmax(logits, dim=2)    # B, T, V, 1
        
        if k_gt is not None:                  # B, V or B, T, V
            # Ensure k_gt is float
            if k_gt.dtype != weights.dtype:
                k_gt = k_gt.float()
            
            # Handle different k_gt dimensions
            if k_gt.dim() == 1:  # V
                raise ValueError("k_gt must have batch dimension.")
            if k_gt.dim() == 2:  # B, V
                # Expand k_gt to match weights dimensions: B, T, V, 1
                k_gt = k_gt.unsqueeze(1).unsqueeze(-1)  # B, 1, V, 1
                k_gt = k_gt.expand_as(weights)  # B, T, V, 1
            elif k_gt.dim() == 3:  # B, T, V
                # Add channel dimension
                k_gt = k_gt.unsqueeze(-1)  # B, T, V, 1
            
            # Normalize k_gt to distribution if it contains binary values
            if torch.any((k_gt == 0) | (k_gt == 1)):
                # Sum over joint dimension (dim=2)
                denom = k_gt.sum(dim=2, keepdim=True).clamp(min=1e-8)
                k_gt = k_gt / denom
            
            # Calculate calibration loss
            loss_calib = self.mse_loss(weights, k_gt)
            return weights, loss_calib
        
        return weights, None
