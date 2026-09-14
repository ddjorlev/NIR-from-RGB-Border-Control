import torch
import torch.nn as nn
from typing import Literal, List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BackboneOutput:
    """
    Standardized output for Backbones.
    """

    embedding: (
        torch.Tensor
    )  # The main global vector (B, D) - Used for Recognition, normalized
    norm: torch.Tensor  # L2 norm of embedding (B, 1)
    feature_maps: Dict[str, torch.Tensor] = field(
        default_factory=dict
    )  # Intermediate layers {"layer2": (B, C, H, W), ...}

    def __post_init__(self):
        # Ensure embeddings are normed
        embs = self.embedding
        norms = torch.norm(embs, p=2, dim=-1, keepdim=True)

        # Ensure embeddings are normed (norm1) or 0 (for padding)
        # norms have to be either 1 (for valid embeddings) or 0 (for padded items), but not something in between
        assert torch.all(
            (torch.isclose(norms, torch.ones_like(norms), atol=1e-5))
            | (torch.isclose(norms, torch.zeros_like(norms), atol=1e-5))
        ), "Embeddings must be L2-normalized (unit norm) or zero for padding"


class BaseBackbone(nn.Module, ABC):
    """
    Abstract Base Class for all Backbones (ResNet, Swin, ViT, etc.).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_weights(
        self,
        weights_path: Optional[str] = None,
        weights_state_dict: Optional[Dict] = None,
    ):
        pass

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> BackboneOutput:
        """
        Must return a BackboneOutput object.
        If the model only produces an embedding, return BackboneOutput(embedding=emb).
        """
        pass

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        return self.forward_features(x)


class _IResNetBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        downsample_mode: Literal["conv", "maxpool"] = "conv",
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.downsample = self._make_downsample(in_ch, out_ch, stride, downsample_mode)

    def _make_downsample(self, in_ch, out_ch, stride, mode):
        if mode not in ("conv", "maxpool"):
            raise ValueError(f"Unknown downsample_mode '{mode}'")
        if stride == 1 and in_ch == out_ch:
            return nn.Identity()
        if mode == "maxpool" and stride != 1 and in_ch == out_ch:
            return nn.MaxPool2d(kernel_size=1, stride=stride)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        skip = self.downsample(x)
        y = self.bn1(x)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.prelu(y)
        y = self.conv2(y)
        y = self.bn3(y)
        return y + skip


class _IResNetBackbone(BaseBackbone):
    BASE_DIM = 64
    INPUT_H, INPUT_W = 112, 112

    def __init__(
        self,
        feature_dim: int,
        blocks_per_layer: List[int],
        dropout_p: float = 0,
        downsample_mode: Literal["conv", "maxpool"] = "conv",
        feature_bn_affine: bool = True,
        return_nodes: Optional[Dict[str, str]] = None,
        weights_path: Optional[str] = None,
        weights_state_dict: Optional[Dict] = None,
    ):
        super().__init__()
        self._out_features = feature_dim
        # Save the requested nodes to intercept natively during forward pass
        self.return_nodes = return_nodes

        self.in_conv = nn.Conv2d(3, self.BASE_DIM, 3, 1, 1, bias=False)
        self.in_bn = nn.BatchNorm2d(self.BASE_DIM)
        self.in_prelu = nn.PReLU(self.BASE_DIM)

        self.layers = nn.ModuleList()
        in_ch = self.BASE_DIM
        out_chs = [64 << i for i in range(len(blocks_per_layer))]

        for _, (num_blocks, out_ch) in enumerate(zip(blocks_per_layer, out_chs)):
            layer_seq = self._make_layer(in_ch, out_ch, num_blocks, downsample_mode)
            self.layers.append(layer_seq)
            in_ch = out_ch

        final_dim = in_ch
        final_h = self.INPUT_H >> len(self.layers)
        final_w = self.INPUT_W >> len(self.layers)

        self.out_bn = nn.BatchNorm2d(final_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(final_dim * final_h * final_w, feature_dim, bias=True)
        self.feature_bn = nn.BatchNorm1d(feature_dim, affine=feature_bn_affine)

        if weights_path is not None or weights_state_dict is not None:
            self.load_weights(
                weights_path=weights_path, weights_state_dict=weights_state_dict
            )

    def load_weights(
        self,
        weights_path: Optional[str] = None,
        weights_state_dict: Optional[Dict] = None,
    ):
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
            print(f"[{self.__class__.__name__}] Weights loaded from path.")
        elif weights_state_dict is not None:
            self.load_state_dict(weights_state_dict, strict=True)
            print(f"[{self.__class__.__name__}] Weights loaded from state dict.")

    def _make_layer(self, in_channels, out_channels, num_blocks, downsample_mode):
        blocks = []
        block = _IResNetBlock(in_channels, out_channels, 2, downsample_mode)
        blocks.append(block)
        for _ in range(1, num_blocks):
            block = _IResNetBlock(out_channels, out_channels, 1, downsample_mode)
            blocks.append(block)
        return nn.Sequential(*blocks)

    @property
    def out_features(self) -> int:
        return self._out_features

    # Rename this from 'forward' to 'forward_features' to satisfy BaseBackbone
    def forward_features(self, x: torch.Tensor) -> BackboneOutput:
        feature_maps = {} if self.return_nodes else None

        x = self.in_conv(x)
        x = self.in_bn(x)
        x = self.in_prelu(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Natively intercept and map requested layers
            if self.return_nodes:
                node_name = f"layers.{i}"
                if node_name in self.return_nodes:
                    out_name = self.return_nodes[node_name]
                    feature_maps[out_name] = x

        x = self.out_bn(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.feature_bn(x)

        norm = torch.norm(x, p=2, dim=-1, keepdim=False)
        x = torch.div(x, norm.unsqueeze(-1))

        return BackboneOutput(embedding=x, norm=norm, feature_maps=feature_maps)

    # Add the standard forward wrapper back in
    def forward(self, x: torch.Tensor) -> BackboneOutput:
        return self.forward_features(x)


class IResNet18Backbone(_IResNetBackbone):
    def __init__(
        self,
        dropout_p=0,
        downsample_mode="conv",
        feature_bn_affine=False,
        return_nodes=None,
        weights_path: Optional[str] = None,
        weights_state_dict: Optional[Dict] = None,
    ):
        super().__init__(
            feature_dim=512,
            blocks_per_layer=[2, 2, 2, 2],
            dropout_p=dropout_p,
            downsample_mode=downsample_mode,
            feature_bn_affine=feature_bn_affine,
            return_nodes=return_nodes,
            weights_path=weights_path,
            weights_state_dict=weights_state_dict,
        )


class IResNet34Backbone(_IResNetBackbone):
    def __init__(
        self,
        dropout_p=0,
        downsample_mode="conv",
        feature_bn_affine=False,
        return_nodes=None,
        weights_path: Optional[str] = None,
        weights_state_dict: Optional[Dict] = None,
    ):
        super().__init__(
            feature_dim=512,
            blocks_per_layer=[3, 4, 6, 3],
            dropout_p=dropout_p,
            downsample_mode=downsample_mode,
            feature_bn_affine=feature_bn_affine,
            return_nodes=return_nodes,
            weights_path=weights_path,
            weights_state_dict=weights_state_dict,
        )


class IResNet50Backbone(_IResNetBackbone):
    def __init__(
        self,
        dropout_p=0,
        downsample_mode="conv",
        feature_bn_affine=False,
        return_nodes=None,
        weights_path: Optional[str] = None,
        weights_state_dict: Optional[Dict] = None,
    ):
        super().__init__(
            feature_dim=512,
            blocks_per_layer=[3, 4, 14, 3],
            dropout_p=dropout_p,
            downsample_mode=downsample_mode,
            feature_bn_affine=feature_bn_affine,
            return_nodes=return_nodes,
            weights_path=weights_path,
            weights_state_dict=weights_state_dict,
        )


class IResNet100Backbone(_IResNetBackbone):
    def __init__(
        self,
        dropout_p=0,
        downsample_mode="conv",
        feature_bn_affine=False,
        return_nodes=None,
        weights_path: Optional[str] = None,
        weights_state_dict: Optional[Dict] = None,
    ):
        super().__init__(
            feature_dim=512,
            blocks_per_layer=[3, 13, 30, 3],
            dropout_p=dropout_p,
            downsample_mode=downsample_mode,
            feature_bn_affine=feature_bn_affine,
            return_nodes=return_nodes,
            weights_path=weights_path,
            weights_state_dict=weights_state_dict,
        )


class IResNet200Backbone(_IResNetBackbone):
    def __init__(
        self,
        dropout_p=0,
        downsample_mode="conv",
        feature_bn_affine=False,
        return_nodes=None,
        weights_path: Optional[str] = None,
        weights_state_dict: Optional[Dict] = None,
    ):
        super().__init__(
            feature_dim=512,
            blocks_per_layer=[6, 26, 60, 6],
            dropout_p=dropout_p,
            downsample_mode=downsample_mode,
            feature_bn_affine=feature_bn_affine,
            return_nodes=return_nodes,
            weights_path=weights_path,
            weights_state_dict=weights_state_dict,
        )


class TransformNormalizePreprocessor(nn.Module):
    def __init__(
        self,
        mean: List[float],
        std: List[float],
        color_space: str,
        transform: Optional[nn.Module] = None,
        do_range_check: bool = True,  # Set to False for max speed in production
    ):
        super().__init__()
        # Register as buffers so they move to GPU automatically but aren't trainable
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

        self.transform = transform if transform is not None else nn.Identity()
        self.color_space = color_space
        self.do_range_check = do_range_check

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4  # (B, C, H, W)
        # Optimization: Only check range if requested (avoids GPU sync)
        if self.do_range_check:
            # We use a loose epsilon for floating point errors
            if x.min() < -0.01 or x.max() > 1.01:
                # Raise error or warning depending on strictness preference
                raise ValueError(
                    f"Input images expected in [0, 1], but got range [{x.min():.2f}, {x.max():.2f}]. "
                    "Normalization is model-dependent."
                )

        match self.color_space:
            case "BGR":
                x = x.flip(1)  # RGB to BGR

        x = self.transform(x)
        x_normalized = (x - self.mean) / self.std
        return x_normalized


def get_adaface_ir101_model(weights_path: str = "converted_adaface_ir101_ms1mv3.pt"):
    model = IResNet100Backbone(
        downsample_mode="maxpool",
        feature_bn_affine=False,
        weights_path=weights_path,
    )
    model.eval()
    preprocessor = TransformNormalizePreprocessor(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        transform=None,
        color_space="BGR",
    )
    return model, preprocessor, "adaface_ir101_ms1mv3"


def get_arcface_ir101_model(weights_path: str = "converted_arcface_r100_ms1mv3.pt"):
    model = IResNet100Backbone(
        downsample_mode="conv",
        feature_bn_affine=False,
        weights_path=weights_path,
    )
    model.eval()
    preprocessor = TransformNormalizePreprocessor(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        transform=None,
        color_space="RGB",
    )
    return model, preprocessor, "arcface_ir101_ms1mv3"
