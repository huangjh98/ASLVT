import math
import torch
import copy
import einops
import torch.nn.functional as F
from torch import nn

from dataclasses import dataclass
from typing import Optional
from enum import IntEnum
from einops import rearrange



import math
import warnings
import torch
import torch.nn.functional as F
from torch import Tensor, nn


eps_fea_norm = 1e-5
eps_l2_norm = 1e-10


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x: Tensor, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor):
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor):
        return drop_path(x, self.drop_prob, self.training)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor):
        return F.gelu(input)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        nn.init.constant_(self.proj.weight.data, 0.0)
        nn.init.constant_(self.proj.bias.data, 0.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        B_q, N_q, _ = q.size()
        B_k, N_k, _ = k.size()
        q = self.q(q).reshape(B_q, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.attn_drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        q = (attn @ v).transpose(1, 2).reshape(q.size(0), q.size(2), -1)
        q = self.proj_drop(self.proj(q))
        return q


class Encoder(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn = nn.BatchNorm1d(dim)
        self.mlp = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, n, d = x.size()
        x = x + self.drop_path(self.attn(x, x, x))
        x_bn = self.bn(x.reshape(b * n, d)).reshape(b, n, d)
        x = x + self.drop_path(self.mlp(x_bn))
        return x


class Decoder1(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.self_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=2 * dim, out_features=dim, drop=drop)

    def forward(self, q, x):
        q_bn = q#self.bn1(q)
        
        q = x + self.drop_path(self.cross_attn(q_bn, x, x))
        
        q = q+ self.drop_path(self.mlp(q))
        q_bn = self.bn2(q)
        q = q + self.drop_path(self.self_attn(q_bn, q_bn, q_bn))
 
        return q





def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid

    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.k = k

    def __call__(self, x, sigma):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        # print('x', x.shape)
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma  # b, nS, d
        #print('perturbed_x', perturbed_x.shape)
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        #print('topk_results',topk_results.indices.shape)

        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k
        #print('indices', indices.shape ,indices[0,0,0])

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)  # b, k, d
        #print(indicators)
        #print('perturbed_output', perturbed_output.shape, perturbed_output[0,indices[0,0,0],0,0])

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                    torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                    / ctx.num_samples
                    / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)

class PerturbedTopKFunction1(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, k: int, num_samples: int = 512, sigma: float = 0.05):
        # print('x', x.shape)
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        #noise = torch.normal(mean=0.0, std=1.0, size=(b, d, num_samples)).to(x.device)
        noise = torch.zeros(b, d, num_samples).to(x.device)
        
        sorted_tensor, _ = torch.sort(x, dim=-1, descending=True)
        start_idx = d * 3 // 4
        
        std, _ = torch.std_mean(x,-1)
        mean = sorted_tensor[:, start_idx]
        
        x = (x - mean.unsqueeze(1).repeat(1, d)) / (std.unsqueeze(1).repeat(1, d) ** 0.3)
        
        x = sigmoid(x, 0.001)
        perturbed_x = x[:, :, None] + noise * sigma  # b, nS, d

        return perturbed_x

    

def HardTopK(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices  # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def extract_frames_from_indices(x, indices):
    batch_size, _, channels = x.shape
    k = indices.shape[-1]
    all_frame = x
    frames = batched_index_select(all_frame, 1, indices)
    frames = frames.contiguous().view(batch_size, k, channels)
    return frames


def extract_frames_from_indicators(x, indicators):
    indicators = rearrange(indicators, "b d k -> b k d")
    frames = torch.einsum("b k d, b d c-> b k c", indicators, x)
    return frames


class ModalityEmbeddingsID(IntEnum):
    TEXT_QUESTION = 0
    TEXT_EMBEDDING = 1
    TEXT_UNUSED = 2  # ignore
    VISUAL_EMBEDDING = 3
    VISUAL_UNUSED = 4  # ignore


class ModalityEmbeddings(nn.Module):
    """
    Provides embeddings that indicate type of modality; for use with multimodal inputs for ATP. See atp.py for usage.
    """

    def __init__(self,
                 d_model: int,
                 use_text_query: bool = False,
                 use_text_cands: bool = False,
                 n_cands: int = 5):
        """
        Details for each of these arguments are provided in ATPConfig.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=len(ModalityEmbeddingsID),
                                      embedding_dim=d_model)

        self.use_text_query = use_text_query
        self.use_text_cands = use_text_cands
        self.n_cands = n_cands if use_text_cands else 0
        self.n_text_feats = 1 if use_text_query else 0
        if use_text_cands:
            self.n_text_feats += n_cands

    def forward(self, x, num_frame):
        """
        x: torch.tensor of size (L, N, D)
        returns modality embeddings for x of size (L, *, D)
        """
        L, N, D = x.size()  # (sequence_length, batch_size, feature_dim)
        num_txt = L - num_frame

        # assemble the IDs for the modality encodings, language inputs then vision inputs
        class_ids = []
        if self.use_text_query:
            class_ids.extend([ModalityEmbeddingsID.TEXT_QUESTION, ] * num_txt)
        # if self.use_text_cands:
        #     class_ids.extend([ModalityEmbeddingsID.TEXT_EMBEDDING,] * self.n_cands)
        class_ids.extend([ModalityEmbeddingsID.VISUAL_EMBEDDING, ] * num_frame)

        class_ids = torch.tensor(
            class_ids,
            dtype=torch.long,
            device=x.device
        ).unsqueeze(-1)

        # return modality embeddings
        return self.embedding(class_ids)


@dataclass
class ATPConfig:
    '''
    ATPConfig contains the parameters needed for the ATPSelectorModel (and its ATPEncoder).
    '''
    # ATPEncoder params
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 256
    d_input_t: int = 512
    d_input_v: int = 512
    d_model_ff: int = 256
    enc_dropout: float = 0.1
    use_text_query: bool = True  # at least one use_text_* needs to be true for ATP to be multimodal
    use_text_cands: bool = False  # ^ see above. (note: if both are false, ATP is vision-only)
    n_cands: int = 5  # only relevant when use_text_cands is set to true
    # ATPSelector params
    use_ste: bool = True  # controls type of selector during ATP training; see ATPSelectorModel.forward
    sel_dropout: float = 0.0
    d_input: int = 512  # size of the input vision-language embeddings (e.g. CLIP-ViT-B32 is size 512)

    def default_args(cls):
        return cls(n_layers=6,
                   n_heads=4,
                   d_model=256,
                   d_input_t=512,
                   d_input_v=512,
                   d_model_ff=256,
                   enc_dropout=0.1,
                   use_text_query=True,
                   use_text_cands=False,
                   n_cands=5,
                   use_ste=True,
                   sel_dropout=0.0,
                   d_input=512)

    @classmethod
    def from_args(cls, args):
        return cls(n_layers=args.n_layers,
                   n_heads=args.n_heads,
                   d_model=args.d_model,
                   d_model_ff=args.d_model_ff,
                   enc_dropout=args.enc_dropout,
                   use_text_query=args.use_text_query,
                   use_text_cands=args.use_text_cands,
                   n_cands=args.n_cands,
                   use_ste=args.use_ste,
                   sel_dropout=args.sel_dropout,
                   d_input=args.d_input)

@dataclass
class ATPConfig1:
    '''
    ATPConfig contains the parameters needed for the ATPSelectorModel (and its ATPEncoder1).
    '''
    # ATPEncoder1 params
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 512
    d_input_t: int = 512
    d_input_v: int = 512
    d_model_ff: int = 256
    enc_dropout: float = 0.1
    use_text_query: bool = True  # at least one use_text_* needs to be true for ATP to be multimodal
    use_text_cands: bool = False  # ^ see above. (note: if both are false, ATP is vision-only)
    n_cands: int = 5  # only relevant when use_text_cands is set to true
    # ATPSelector params
    use_ste: bool = True  # controls type of selector during ATP training; see ATPSelectorModel.forward
    sel_dropout: float = 0.0
    d_input: int = 512  # size of the input vision-language embeddings (e.g. CLIP-ViT-B32 is size 512)

    def default_args(cls):
        return cls(n_layers=2,
                   n_heads=4,
                   d_model=512,
                   d_input_t=512,
                   d_input_v=512,
                   d_model_ff=256,
                   enc_dropout=0.1,
                   use_text_query=True,
                   use_text_cands=False,
                   n_cands=5,
                   use_ste=True,
                   sel_dropout=0.0,
                   d_input=512)

    @classmethod
    def from_args(cls, args):
        return cls(n_layers=args.n_layers,
                   n_heads=args.n_heads,
                   d_model=args.d_model,
                   d_model_ff=args.d_model_ff,
                   enc_dropout=args.enc_dropout,
                   use_text_query=args.use_text_query,
                   use_text_cands=args.use_text_cands,
                   n_cands=args.n_cands,
                   use_ste=args.use_ste,
                   sel_dropout=args.sel_dropout,
                   d_input=args.d_input)

class ATPEncoder(nn.Module):
    """
    The multimodal transformer encoder for the ATP model. For analysis purposes, the ATP encoder
    does not use any positional information (no positional encodings + transformer / self-attention)
    and is generally kept low-capacity. If the goal is raw accuracy (not analysis), you can relax these constraints.
    """

    def __init__(self, config: ATPConfig):
        """
        config: ATPConfig with parameters for the (transformer-based, atemporal) encoder for ATP.
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.d_model = config.d_model

        self.dropout = nn.Dropout(p=config.enc_dropout)

        self.modality_encoding = ModalityEmbeddings(d_model=self.d_model,
                                                    use_text_query=config.use_text_query,
                                                    use_text_cands=config.use_text_cands,
                                                    n_cands=config.n_cands)

        atp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(atp_encoder_layer, config.n_layers)

    def forward(self, x_inputs: torch.tensor, vis_L):
        """
        x_inputs: torch.tensor of shape (L, N, D)
        """
        L, N, D = x_inputs.size()  # (sequence_length, batch_size, d_model)
        assert D == self.d_model, "inputs dimension mismatch"
        x_encoded = x_inputs * math.sqrt(self.d_model)
        x_encoded += self.modality_encoding(x_encoded, vis_L)
        x_encoded = self.dropout(x_encoded)
        x_encoded = self.transformer_encoder(x_encoded)

        return x_encoded

class ATPEncoder1(nn.Module):
    """
    The multimodal transformer encoder for the ATP model. For analysis purposes, the ATP encoder
    does not use any positional information (no positional encodings + transformer / self-attention)
    and is generally kept low-capacity. If the goal is raw accuracy (not analysis), you can relax these constraints.
    """

    def __init__(self, config: ATPConfig1):
        """
        config: ATPConfig with parameters for the (transformer-based, atemporal) encoder for ATP.
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.d_model = config.d_model

        self.dropout = nn.Dropout(p=config.enc_dropout)

        self.modality_encoding = ModalityEmbeddings(d_model=self.d_model,
                                                    use_text_query=config.use_text_query,
                                                    use_text_cands=config.use_text_cands,
                                                    n_cands=config.n_cands)

        atp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(atp_encoder_layer, config.n_layers)

    def forward(self, x_inputs: torch.tensor, vis_L):
        """
        x_inputs: torch.tensor of shape (L, N, D)
        """
        L, N, D = x_inputs.size()  # (sequence_length, batch_size, d_model)
        assert D == self.d_model, "inputs dimension mismatch"
        x_encoded = x_inputs * math.sqrt(self.d_model)
        x_encoded += self.modality_encoding(x_encoded, vis_L)
        x_encoded = self.dropout(x_encoded)
        x_encoded = self.transformer_encoder(x_encoded)

        return x_encoded

class TopK_Selector(nn.Module):
    """
    The Atemporal Probe (ATP) selector model. Takes as input a sequence of image-language
    encoding and outputs a (discrete) selection over the input frames, to help analyze
    downstream discriminative video-language tasks.
    """

    def __init__(self, config=ATPConfig, num_select=4):
        """
        config: ATPConfig with parameters for initializing the ATPSelectorModel (and its encoder).
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.config = config
        self.t_embedding = nn.Linear(config.d_input_t, config.d_input)
        self.v_embedding = nn.Linear(config.d_input_v, config.d_input)
        self.embedding = nn.Linear(config.d_input, config.d_model)
        self.atp_encoder = ATPEncoder(config)
        self.dropout = nn.Dropout(p=config.sel_dropout)
        self.logits = nn.Linear(config.d_model, 1)
        self.num_select = num_select
        self.sigma = 0.1
        #self.PerturbedTopKFunction = PerturbedTopKFunction()
        self.atp_encoder1 = ATPEncoder1(ATPConfig1)
        self.tokens = nn.Sequential(nn.Conv1d(in_channels=12, out_channels=8, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(8))
        self.token_norm = nn.LayerNorm(12)
        self.cossatt = nn.MultiheadAttention(embed_dim=512, num_heads=4)

    def forward(self,
                x_vis,  # [b, t, d]
                x_txt,  # [b, n, d]
                **kwargs):
        """
        """
        x_vis_cls = x_vis[:, :, :]  # b t n c
        N, vis_L, D = x_vis_cls.size()  # (batch_size, sequence_length, feature_dimension)
        # embed the input sequence to the (smaller) model dimension (d_model) with modality encodings.
        x_vis_cls = self.v_embedding(self.dropout(x_vis_cls))
        x_txt = self.t_embedding(self.dropout(x_txt))
        
        x_vis_cls1 = x_vis_cls
        
        x_inputs = []
        x_vis_cls = x_vis_cls.permute(1, 0, 2)
        x_inputs.append(x_txt.permute(1, 0, 2))  # (n, b, d)
        x_inputs.append(x_vis_cls)
        x_inputs = torch.cat(x_inputs, dim=0)
        x_encoded = self.embedding(self.dropout(x_inputs))
        x_atp_encoded = self.atp_encoder(x_encoded, vis_L)
        x_atp_encoded = x_atp_encoded.permute(1, 0, 2)
        
        x_encoded_v = x_atp_encoded[:, -vis_L:, :]
        x_logits = self.logits(self.dropout(x_encoded_v)).squeeze()
        #indices = self.PerturbedTopKFunction(x_logits, self.num_select)
        

        #qa_frames = x_vis * indices
        #qa_frames = self.atp_encoder1(qa_frames, vis_L)
        if True:#self.training:
            indices = PerturbedTopKFunction.apply(x_logits, self.num_select)
            indices = einops.rearrange(indices, "b k d -> b d k")
            qa_frames = extract_frames_from_indicators(x_vis, indices)
            
        
        return qa_frames
        
class TopK_Selector1(nn.Module):
    """
    The Atemporal Probe (ATP) selector model. Takes as input a sequence of image-language
    encoding and outputs a (discrete) selection over the input frames, to help analyze
    downstream discriminative video-language tasks.
    """

    def __init__(self, config=ATPConfig, num_select=4):
        """
        config: ATPConfig with parameters for initializing the ATPSelectorModel (and its encoder).
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.config = config
        self.t_embedding = nn.Linear(config.d_input_t, config.d_input)
        self.v_embedding = nn.Linear(config.d_input_v, config.d_input)
        self.embedding = nn.Linear(config.d_input, config.d_model)
        self.atp_encoder = ATPEncoder(ATPConfig1)
        self.dropout = nn.Dropout(p=config.sel_dropout)
        self.logits = nn.Linear(config.d_model, 1)
        self.num_select = num_select
        self.sigma = 0.1
        #self.PerturbedTopKFunction = PerturbedTopKFunction()
        self.atp_encoder1 = ATPEncoder1(ATPConfig1)
        
        self.decoder = nn.ModuleList([Decoder1(dim=512, num_heads=4, qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1) for _ in range(2)])
        self.encoder = nn.ModuleList([Encoder(dim=512, num_heads=4, qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1) for _ in range(2)])
        self.tokens = nn.Sequential(nn.Conv1d(in_channels=12, out_channels=8, kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(8))
        self.token_norm = nn.LayerNorm(12)
        self.cossatt = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.dropout = nn.Dropout(p=0.2)
        self.norm = nn.LayerNorm(8)
        self.linear = nn.Linear(512, 512)

    def forward(self,
                x_vis,  # [b, t, d]
                **kwargs):
        """
        """
        x_vis_cls = x_vis[:, :, :]  # b t n c
        N, vis_L, D = x_vis_cls.size()  # (batch_size, sequence_length, feature_dimension)
        # embed the input sequence to the (smaller) model dimension (d_model) with modality encodings.
        x_atp_encoded = self.v_embedding(self.dropout(x_vis_cls))#.permute(0,2,1)
        
        for encoder in self.encoder:
            x_atp_encoded = encoder(x_atp_encoded)
        
        #x_atp_encoded = x_atp_encoded.permute(0,2,1)
        x_atp_encoded1 = self.tokens(x_atp_encoded)
        attns = F.softmax(x_atp_encoded1, dim=1)
        
        q = self.token_norm(torch.mean(attns.unsqueeze(2) * x_atp_encoded.unsqueeze(1), dim=-1))
        
        q = F.softmax(q, dim=1)
        qa_frames = self.linear((q @ x_atp_encoded) + x_atp_encoded1) + x_atp_encoded1
        
        return qa_frames
        


if __name__ == "__main__":
    selector_config = ATPConfig.default_args

    Selector = TopK_Selector(num_select=8)  # .eval()

    #x_vis = torch.rand([2, 8, 257, 1408])
    #x_txt = torch.rand([2, 68, 2048])

    x_vis = torch.rand([32, 12, 512])
    x_txt = torch.rand([32, 32, 512])

    out1,out2 = Selector(x_vis, x_txt)
    print(out1.shape, out2.shape)
