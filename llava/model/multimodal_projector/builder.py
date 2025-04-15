import torch
import torch.nn as nn
import re


from functools import partial
# from .ms_cross_attn import MSCrossAttnBlock
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage



def build_pos_embeds(
    pos_emb,num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if pos_emb:
        pos_emb1 = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb1, mean=0.0, std=0.02)
    else:
        pos_emb1 = None

    return pos_emb1


def build_eos_tokens(num_eos_tokens,initializer_range,output_hidden_size: int):
    # think tokens

    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(prenorm,encoder_hidden_size):
    if prenorm:
        prenorm1 = LayerNorm(encoder_hidden_size)
    else:
        prenorm1 = None
    return prenorm1


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, vision_tower, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_depth = 2
    if config.layer_fusing_strategy =='E_D':
        if config.layer_using_strategy == '18' or config.layer_using_strategy == 'former' or config.layer_using_strategy == 'latter':
            modules = [nn.Linear(config.mm_hidden_size*2, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)      
        if config.layer_using_strategy == '3-18-23' or config.layer_using_strategy == 'all':
            modules = [nn.Linear(config.mm_hidden_size*3, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)  
        if config.layer_using_strategy == 'last':
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)
        
    # elif config.layer_fusing_strategy =='E_M':
    #     # External module fusion strategy:
    #     # - For multi-layer visual features, we apply the MMFuser (Cao et al., 2024) method.
    #     # - After the MMFuser step, the fused features are passed through a two-layer MLP (projector) for further transformation.
    #     # Reference:
    #     # Cao, Y., Liu, Y., Chen, Z., et al. (2024). 
    #     # "MMFuser: Multimodal Multi-Layer Feature Fuser for Fine-Grained Vision-Language Understanding." 
    #     # arXiv preprint arXiv:2410.11829. 
    #     # Available at: https://arxiv.org/abs/2410.11829

    #     if config.layer_using_strategy == '18':
    #         modules = [MSCrossAttnBlock(n_levels=1,d_model=config.mm_hidden_size)]
    #     if config.layer_using_strategy == '3-18-23':
    #         modules = [MSCrossAttnBlock(n_levels=2,d_model=config.mm_hidden_size)]
    #     if config.layer_using_strategy == 'former' or config.layer_using_strategy == 'latter':
    #         modules = [MSCrossAttnBlock(n_levels=12,d_model=config.mm_hidden_size)]   
    #     if config.layer_using_strategy == 'all':
    #         modules = [MSCrossAttnBlock(n_levels=24,d_model=config.mm_hidden_size)]   
    #     if  config.layer_using_strategy == '3-18': 
    #         return  # 3-18 and 3-18-23 are consistent in External fusion
    #     modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
    #     for _ in range(1, mlp_depth):
    #         modules.append(nn.GELU())
    #         modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    #     return nn.Sequential(*modules)
    
    else:
        # projectors in Internal fusion 
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)