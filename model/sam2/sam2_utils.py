import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class TemporalPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=100):
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def forward(self, seq_len, device):
        ids = torch.arange(seq_len, device=device)
        return self.emb(ids)  # [seq_len, dim]

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=2*3.1415):
        super().__init__()
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).float()
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).float()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1)  # C x H x W
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # B x C x H x W
        return pos

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ObjPointer(nn.Module):
    def __init__(self, mlp_dim,hidden_dim, num_classes, use_position_encoding=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_position_encoding = use_position_encoding

        # MLP for non-linear feature aggregation
        self.mlp = MLP(mlp_dim, mlp_dim//4, hidden_dim, 3)  # Non-linear aggregation of features

        if self.use_position_encoding:
            self.position_encoding = nn.Parameter(
                torch.randn(1, num_classes, hidden_dim))  # Learnable position encoding


    def forward(self, low_res_features):
        """
        Forward pass for calculating object pointers.

        Args:
        - low_res_features: Tensor of shape [B, num_classes, H*W]
        - high_res_features: Tensor of shape [B, num_classes, H*W] (optional)
        - spatial_pos: Tensor of shape [B, H*W, C] containing spatial position encodings (optional)

        Returns:
        - obj_ptr: Tensor of shape [B, num_classes, C], representing the object pointers
        """


        # Step 2: Apply a non-linear MLP to combine the features
        obj_ptr = self.mlp(low_res_features)

        # Step 3: Optionally add position encoding
        if self.use_position_encoding:
            obj_ptr = obj_ptr + self.position_encoding


        # Final shape: [B, num_classes, C] (Object pointers for each class)
        return obj_ptr

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def load_pretrained_weights(model, pretrained_path, strict=False):
    """
    加载预训练权重并检查形状是否匹配，不匹配的权重将被跳过。

    Args:
        model (nn.Module): 目标模型。
        pretrained_path (str): 预训练权重文件路径。
        strict (bool): 是否严格要求加载所有权重，默认为 False。

    Returns:
        model (nn.Module): 更新后的模型。
    """
    # 加载预训练权重
    state_dict = torch.load(pretrained_path, map_location='cpu')

    # 获取当前模型的权重字典
    model_dict = model.state_dict()

    # 过滤出在当前模型字典中存在的键
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # 记录需要删除的键
    keys_to_remove = []

    # 检查形状是否匹配，并只更新匹配的权重
    for k, v in pretrained_dict.items():
        if v.shape != model_dict[k].shape:
            # print(f"Skipping weight {k} due to shape mismatch: "
            #       f"pretrained shape {v.shape} vs model shape {model_dict[k].shape}")
            # 记录不匹配的键
            keys_to_remove.append(k)

    # 删除不匹配的权重
    for key in keys_to_remove:
        pretrained_dict.pop(key)

    # 更新模型权重
    model_dict.update(pretrained_dict)

    # 加载更新后的权重到模型
    model.load_state_dict(model_dict, strict=strict)

    return model

class CircularList:
    def __init__(self, max_length):
        """初始化列表类，指定最大长度"""
        self.max_length = max_length
        self.data = []

    def append(self, item):
        self.data.append(item)
        if len(self.data) > self.max_length:
            self.data.pop(0)  # 移除第一个元素（最先加入的数据）

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.data = []