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
        return cls(n_layers=6,
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


class Selector(nn.Module):
    """
    The Atemporal Probe (ATP) selector model. Takes as input a sequence of image-language
    encoding and outputs a (discrete) selection over the input frames, to help analyze
    downstream discriminative video-language tasks.
    """

    def __init__(self, config=ATPConfig):
        """
        config: ATPConfig with parameters for initializing the ATPSelectorModel (and its encoder).
        See ATPConfig documentation for details.
        """
        super().__init__()
        self.config = config
        #self.t_embedding = nn.Linear(config.d_input_t, config.d_input)
        #self.v_embedding = nn.Linear(config.d_input_v, config.d_input)
        self.embedding = nn.Linear(config.d_input, config.d_model)
        self.atp_encoder = ATPEncoder(config)
        self.dropout = nn.Dropout(p=config.sel_dropout)

    def forward(self,
                x_vis,  # [b, t, d]
                x_txt,  # [b, n, d]
                **kwargs):
        """
        """
        x_vis_cls = x_vis[:, :, :]  # b t n c
        N, vis_L, D = x_vis_cls.size()  # (batch_size, sequence_length, feature_dimension)
        #x_vis_cls = self.v_embedding(self.dropout(x_vis_cls))
        #x_txt = self.t_embedding(self.dropout(x_txt))
        x_inputs = []
        x_vis_cls = x_vis_cls.permute(1, 0, 2)
        x_inputs.append(x_txt.permute(1, 0, 2))  # (n, b, d)
        x_inputs.append(x_vis_cls)
        x_inputs = torch.cat(x_inputs, dim=0)
        x_encoded = self.embedding(self.dropout(x_inputs))
        x_atp_encoded = self.atp_encoder(x_encoded, vis_L)
        x_atp_encoded = x_atp_encoded.permute(1, 0, 2)
        x_encoded_v = x_atp_encoded[:, -vis_L:, :]
        return x_encoded_v
        


if __name__ == "__main__":
    selector_config = ATPConfig.default_args

    Selector = Selector()  # .eval()

    #x_vis = torch.rand([2, 8, 257, 1408])
    #x_txt = torch.rand([2, 68, 2048])

    x_vis = torch.rand([32, 12, 512])
    x_txt = torch.rand([32, 1, 512])

    out = Selector(x_vis, x_txt)
    print(out.shape)
