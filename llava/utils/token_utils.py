import torch
from transformers import AutoTokenizer

def check_and_reconstruct_identical_tokens(embeddings, tokenizer=None, model=None, print_info=True):
    """
    Check if embeddings are identical and optionally reconstruct tokens.
    
    Args:
        embeddings (torch.Tensor): The embeddings to check
        tokenizer (AutoTokenizer, optional): Tokenizer for reconstruction
        model (nn.Module, optional): Model for getting token embeddings
        print_info (bool): Whether to print debug information
        
    Returns:
        tuple: (are_identical, reconstructed_tokens, decoded_text)
    """
    if embeddings.shape[0] < 2:
        return False, None, None
        
    # Check using gram matrix computation
    gram_matrix = torch.mm(embeddings, embeddings.t())
    gram_diag = torch.diag(gram_matrix)
    gram_off_diag = gram_matrix - torch.diag(gram_diag)
    
    # If all off-diagonal elements are close to diagonal elements, embeddings are identical
    are_identical = torch.allclose(gram_off_diag, gram_diag.unsqueeze(1), rtol=1e-5, atol=1e-5)
    
    if print_info and are_identical and torch.distributed.get_rank() == 0:
        print("Identical text tokens found in gram matrix computation")
        print(f"Gram matrix:\n{gram_matrix}")
        print(f"Text embeds:\n{embeddings}")
    
    if not are_identical or tokenizer is None or model is None:
        return are_identical, None, None
        
    # Reconstruct tokens
    token_embeddings = model.embed_tokens.weight
    distances = torch.cdist(embeddings, token_embeddings)
    closest_token_ids = torch.argmin(distances, dim=1)
    decoded_text = tokenizer.decode(closest_token_ids)
    
    if print_info and torch.distributed.get_rank() == 0:
        print(f"Reconstructed tokens: {closest_token_ids.tolist()}")
        print(f"Decoded text: {decoded_text}")
        
    return are_identical, closest_token_ids, decoded_text 

# def configure_dummy_tokens(model, use_dummy=True, strategy='gaussian', value=0.0):
#     """
#     Helper function to configure dummy image token options for LLaVA models.
    
#     Args:
#         model: The LLaVA model instance
#         use_dummy: Whether to use dummy tokens (True) or real image features (False)
#         strategy: The dummy token strategy to use:
#             - 'gaussian': Gaussian noise with same mean/std as original features
#             - 'zeros': All zeros
#             - 'ones': All ones
#             - 'uniform': Uniform random values between 0 and 1
#             - 'constant': Constant value (specified by value param)
#         value: Value to use for 'constant' strategy
    
#     Returns:
#         The model with updated configuration
#     """
#     # Set configuration options
#     model.config.use_dummy_image_tokens = use_dummy
#     model.config.dummy_token_strategy = strategy
#     model.config.dummy_token_value = value
    
#     # Print configuration for verification
#     print(f"Dummy image tokens: {'Enabled' if use_dummy else 'Disabled'}")
#     if use_dummy:
#         print(f"Strategy: {strategy}")
#         if strategy == 'constant':
#             print(f"Constant value: {value}")
    
#     return model