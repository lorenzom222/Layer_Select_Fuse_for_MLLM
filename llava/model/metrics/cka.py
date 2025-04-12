import torch
from typing import List, Tuple, Optional, Union

def pool_features(features: torch.Tensor, pool_type: str = "mean") -> torch.Tensor:
    """
    Pools the second dimension (patch/token dimension) to produce [B, D].
    
    Args:
        features: A tensor of shape [B, L, D], where:
          - B = batch size
          - L = # of patches (ViT) or # of tokens (LLM)
          - D = embedding dimension
        pool_type: "mean" (average pooling) or "max" (max pooling)
    
    Returns:
        A tensor of shape [B, D], where each sample is pooled along the L dimension.
    """
    if pool_type == "mean":
        # Average across all patches/tokens
        return features.mean(dim=1)  # [B, D]
    elif pool_type == "max":
        # Max across all patches/tokens
        return features.max(dim=1).values  # [B, D]
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

def center_gram(gram: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """
    Centers a symmetric Gram matrix in-place using Kornblith's approach.

    The centering formula (biased version):
        G_centered = G - row_means - col_means + overall_mean
    However, we typically compute row_means == col_means for symmetric G.

    Args:
        gram: A [N, N] symmetric matrix (e.g. from X @ X^T).
        unbiased: Whether to use unbiased centering adjustments (U-statistic).
    
    Returns:
        The centered Gram matrix of shape [N, N].
    """
    if not torch.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")

    gram = gram.clone()
    n = gram.shape[0]

    if unbiased:
        # Follow Szekely & Rizzo's approach, same as in Kornblith's reference code
        gram.fill_diagonal_(0)
        means = torch.sum(gram, dim=0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram = gram - means.unsqueeze(0) - means.unsqueeze(1)
        gram.fill_diagonal_(0)
    else:
        # Biased (simple) centering
        means = torch.mean(gram, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram = gram - means.unsqueeze(0) - means.unsqueeze(1)

    return gram

def cka(gram_x: torch.Tensor, gram_y: torch.Tensor, debiased: bool = False) -> float:
    """
    Compute the scalar CKA value between two centered Gram matrices.
    
    The formula for CKA (biased) is:
        scaled_hsic = sum_{i,j} (K_ij * L_ij)
        norm_x = ||K||_F,   norm_y = ||L||_F
        CKA = scaled_hsic / (norm_x * norm_y)

    where K and L are the centered Gram matrices of X and Y, respectively.
    
    Args:
        gram_x: [N, N] Gram matrix (e.g. x @ x^T).
        gram_y: [N, N] Gram matrix (e.g. y @ y^T).
        debiased: Whether to apply unbiased centering (for HSIC).
    
    Returns:
        A float in [0, 1] indicating the CKA value.
    """
    # 1. Center each Gram matrix
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # 2. Compute HSIC numerator (elementwise dot product of Gram matrices)
    scaled_hsic = torch.sum(gram_x * gram_y)

    # 3. Compute normalization (Frobenius norms of centered Gram matrices)
    norm_x = torch.norm(gram_x)
    norm_y = torch.norm(gram_y)

    # 4. Final CKA ratio
    return (scaled_hsic / (norm_x * norm_y)).item()

def compute_cka_vit_llm(
    visual_features: Union[torch.Tensor, List[torch.Tensor]],
    hidden_states: torch.Tensor,
    pool_type: str = "mean",
    debiased: bool = False
) -> float:
    """
    Compute the CKA between ViT outputs ([B, P, D]) and LLM hidden states ([B, T, D])
    when P != T. We solve the mismatch by pooling each to [B, D].
    
    Steps:
      1. Pool across patch/token dimension => each side becomes [B, D].
      2. Compute Gram matrices => each side becomes [B, B].
      3. Center each Gram matrix, then compute the CKA ratio.

    Formula (biased):
        scaled_hsic = sum_{i,j} (K_ij * L_ij)
        K = (X * X^T) centered, L = (Y * Y^T) centered
        CKA = scaled_hsic / (||K||_F * ||L||_F)

    Args:
        visual_features: [B, P, D] ViT patch embeddings
        hidden_states: [B, T, D] LLM token embeddings
        pool_type: "mean" or "max" for pooling along P or T dimension
        debiased: Whether to apply unbiased HSIC adjustments

    Returns:
        A float in [0, 1], the CKA similarity score.
    """
    if isinstance(visual_features, list):
        # Stack the list of tensors into a single tensor
        visual_features = torch.stack(visual_features)
    
    # Ensure both inputs have the same batch size
    batch_size = min(visual_features.shape[0], hidden_states.shape[0])
    visual_features = visual_features[:batch_size]
    hidden_states = hidden_states[:batch_size]
    
    # 1) Pool each to [B, D]
    vit_pooled = pool_features(visual_features, pool_type=pool_type)  # [B, D]
    llm_pooled = pool_features(hidden_states, pool_type=pool_type)   # [B, D]

    # 2) Construct Gram matrices: shape [B, B]
    gram_v = vit_pooled @ vit_pooled.T
    gram_h = llm_pooled @ llm_pooled.T

    # 3) Compute CKA
    return cka(gram_v, gram_h, debiased=debiased)


# # -------------------------
# # Example usage:
# if __name__ == "__main__":
#     # Suppose we have B=2, P=4, T=3, D=5 (toy data)
#     vit_out = torch.rand(2, 4, 5)  # e.g. from a vision tower
#     llm_out = torch.rand(2, 3, 5)  # e.g. from a language model

#     # Compute CKA
#     score = compute_cka_vit_llm(vit_out, llm_out, pool_type="mean", debiased=False)
#     print("CKA score:", score)



# def compute_layer_cka(
#     visual_features: Union[torch.Tensor, List[torch.Tensor]],
#     hidden_states: torch.Tensor,
#     image_token_mask: Optional[torch.Tensor] = None,
#     pooling_strategy: str = 'all'
# ) -> float:
#     """Compute CKA between visual features and hidden states at a layer.
    
#     Args:
#         visual_features: Visual features from ViT [batch_size, num_patches, hidden_dim] or list of tensors
#         hidden_states: Hidden states from LLM layer [batch_size, seq_len, hidden_dim]
#         image_token_mask: Mask indicating which positions are image tokens
#         pooling_strategy: One of ['all', 'last_e', 'max_pool_last_e', 'avg_pool_last_e', 'max_pool_all', 'avg_pool_all']
        
#     Returns:
#         CKA score between visual features and hidden states
#     """
#     # Handle list of tensors for visual features
#     if isinstance(visual_features, list):
#         # Stack the list of tensors into a single tensor
#         visual_features = torch.stack(visual_features)
    
#     # Convert to numpy for CKA computation
#     v_features = visual_features.detach().cpu().numpy()
#     h_states = hidden_states.detach().cpu().numpy()
    
#     if pooling_strategy == 'all':
#         # Use all positions
#         v_gram = gram_linear(torch.from_numpy(v_features.reshape(-1, v_features.shape[-1])))
#         h_gram = gram_linear(torch.from_numpy(h_states.reshape(-1, h_states.shape[-1])))
        
#     elif pooling_strategy == 'last_e':
#         # Use only last E positions
#         if image_token_mask is not None:
#             mask = image_token_mask.detach().cpu().numpy()
#             h_states = h_states[mask]
#         v_gram = gram_linear(torch.from_numpy(v_features.reshape(-1, v_features.shape[-1])))
#         h_gram = gram_linear(torch.from_numpy(h_states.reshape(-1, h_states.shape[-1])))
        
#     elif pooling_strategy in ['max_pool_last_e', 'avg_pool_last_e']:
#         # Pool last E positions
#         if image_token_mask is not None:
#             mask = image_token_mask.detach().cpu().numpy()
#             h_states = h_states[mask]
#         pool_fn = torch.max if pooling_strategy == 'max_pool_last_e' else torch.mean
#         h_states = pool_fn(torch.from_numpy(h_states), dim=0)
#         v_gram = gram_linear(torch.from_numpy(v_features.reshape(-1, v_features.shape[-1])))
#         h_gram = gram_linear(h_states.unsqueeze(0))
        
#     elif pooling_strategy in ['max_pool_all', 'avg_pool_all']:
#         # Pool all positions
#         pool_fn = torch.max if pooling_strategy == 'max_pool_all' else torch.mean
#         h_states = pool_fn(torch.from_numpy(h_states), dim=0)
#         v_gram = gram_linear(torch.from_numpy(v_features.reshape(-1, v_features.shape[-1])))
#         h_gram = gram_linear(h_states.unsqueeze(0))
        
#     else:
#         raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
#     return cka(v_gram, h_gram) 
