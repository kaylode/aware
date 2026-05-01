import torch
import torch.nn as nn
import pathlib
import os
from .tabpfn.utils import load_model_workflow
from torch.amp import autocast

class KNNPFNv2Model(nn.Module):
    """
    KNNPFNv2 Model Wrapper.
    Loads a TabPFN backbone using load_model_workflow and provides a unified interface
    for classification and regression tasks.
    """
    def __init__(
        self,
        n_out: int,
        ninp: int,
        num_features: int,
        dropout: float = 0.2,
        device: str = "cpu",
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.n_out = n_out
        self.ninp = ninp
        self.num_features = num_features

        # Load TabPFN backbone
        # We assume the tabpfn/utils.py exists or we provide it
        base_path = pathlib.Path(__file__).parent.resolve()
        
        # Load the backbone
        self.backbone_data, self.config, _ = load_model_workflow(
            0, 42, add_name='', 
            base_path=base_path, device=device, 
            only_inference=True
        )
        self.backbone = self.backbone_data[2] # The model is usually the 3rd element
        self.backbone.to(torch.float32)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.num_classes = n_out

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
        task: str = "cls",
        eval_pos: int = None,
        **kwargs
    ):
        """
        Forward pass for KNNPFNv2.
        Args:
            x_context: (K, B, D) or (K, D)
            y_context: (K, B) or (K,)
            x_query: (Q, B, D) or (Q, D)
        """
        if x_context.ndim == 2: # (K, D) -> (K, 1, D)
            x_context = x_context.unsqueeze(1)
            y_context = y_context.unsqueeze(1)
            x_query = x_query.unsqueeze(1)
            
        if eval_pos is None:
            eval_pos = x_context.shape[0]

        # Combine context and query for the transformer
        # TabPFN expects (K+Q, B, D)
        x_full = torch.cat([x_context, x_query], dim=0)
        
        with autocast('cuda', enabled=(x_full.is_cuda)):
            # TabPFN forward: (style, x, y)
            # single_eval_pos specifies where the query starts
            out = self.backbone((None, x_full, y_context), single_eval_pos=eval_pos)
            
        # out shape: (Q, B, num_classes + 1)
        if task == "cls":
            logits = out[:, :, :self.num_classes]
            return logits.to(torch.float32)
        elif task == "reg":
            # Regression often uses the last dimension or a specific head
            # For TabPFN-style regression, it might be the last channel
            reg_out = out[:, :, -1]
            return reg_out.to(torch.float32)
        else:
            raise ValueError(f"Unknown task: {task}")

    @classmethod
    def load(cls, model_state, config, device="cpu"):
        model = cls(
            n_out=config['max_num_classes'],
            ninp=config['emsize'],
            num_features=config['num_features'],
            dropout=config.get('dropout', 0.1),
            device=device
        )
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        return model
