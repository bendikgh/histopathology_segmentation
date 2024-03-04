from monai.losses import DiceLoss
import torch

class DiceLossWrapper(DiceLoss):
    """
    Compute average Dice loss between two tensors.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW, where N is the number of classes.
            target: the shape should be BHW.
        Raises:
            ValueError: If input shape is unequal to BNHW.
            ValueError: If target shape is unequal to BHW.
        """
        if len(input.shape) != 4:
            raise ValueError(f"Input should be of shape BNHW, not {target.shape}")
        if len(target.shape) != 3:
            raise ValueError(f"Target should be of shape BHW, not {target.shape}")
        
        target = target.unsqueeze(1)
        return super().forward(input, target)

