from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast
import torch.nn.functional as F
import numpy as np
import wandb

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
from llava.utils.token_utils import check_and_reconstruct_identical_tokens
from transformers import AutoTokenizer

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
       
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)



# def cosine_similarity(features_x, features_y):

#     cosine_sim = F.cosine_similarity(text_mean, image_mean, dim=0).item()

#     return cosine_sim

# def unbiased_cka(features_x, features_y, pooling: str = 'interpolate'):
#     """
#     Computes the unbiased Centered Kernel Alignment (CKA) similarity between two feature matrices.
#     If the number of samples (rows) does not match, applies pooling (mean or max) to reduce each to a single vector.

#     Args:
#         features_x (torch.Tensor): First feature matrix (n_samples_x, n_features).
#         features_y (torch.Tensor): Second feature matrix (n_samples_y, n_features).

#     Returns:
#         torch.Tensor: The CKA similarity value.
#     """
    
#     return cka_value

# def gram_linear(x):
#     """Compute Gram (kernel) matrix for a linear kernel.

#     Args:
#         x: A num_examples x num_features matrix of features.

#     Returns:
#         A num_examples x num_examples Gram matrix of examples.
#     """
#     return x.dot(x.T)



# def center_gram(gram, unbiased=False):
#     """Center a symmetric Gram matrix.

#     This is equvialent to centering the (possibly infinite-dimensional) features
#     induced by the kernel before computing the Gram matrix.

#     Args:
#         gram: A num_examples x num_examples symmetric matrix.
#         unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
#             estimate of HSIC. Note that this estimator may be negative.

#     Returns:
#         A symmetric matrix with centered columns and rows.
#     """
#     if not np.allclose(gram, gram.T):
#         raise ValueError('Input must be a symmetric matrix.')
#     gram = gram.copy()

#     if unbiased:
#         # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
#         # L. (2014). Partial distance correlation with methods for dissimilarities.
#         # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
#         # stable than the alternative from Song et al. (2007).
#         n = gram.shape[0]
#         np.fill_diagonal(gram, 0)
#         means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
#         means -= np.sum(means) / (2 * (n - 1))
#         gram -= means[:, None]
#         gram -= means[None, :]
#         np.fill_diagonal(gram, 0)
#     else:
#         means = np.mean(gram, 0, dtype=np.float64)
#         means -= np.mean(means) / 2
#         gram -= means[:, None]
#         gram -= means[None, :]

#     return gram

# def cka(gram_x, gram_y, debiased=False):
#     """Compute CKA.

#     Args:
#         gram_x: A num_examples x num_examples Gram matrix.
#         gram_y: A num_examples x num_examples Gram matrix.
#         debiased: Use unbiased estimator of HSIC. CKA may still be biased.

#     Returns:
#         The value of CKA between X and Y.
#     """
#     gram_x = center_gram(gram_x, unbiased=debiased)
#     gram_y = center_gram(gram_y, unbiased=debiased)

#     # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
#     # n*(n-3) (unbiased variant), but this cancels for CKA.
#     scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

#     normalization_x = np.linalg.norm(gram_x)
#     normalization_y = np.linalg.norm(gram_y)
#     return scaled_hsic / (normalization_x * normalization_y)

# def compute_cka(text_embeds, image_embeds):
#     gram_x = gram_linear(text_embeds)
#     gram_y = gram_linear(image_embeds)
#     return cka(gram_x, gram_y)


def gram_linear(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, D]  feature matrix
    returns: [N, N]  Gram matrix = X · Xᵀ
    """
    x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-6)  # [N,D]
    return x @ x.T

def center_gram(gram: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """
    Center a symmetric Gram matrix in PyTorch.
    """
    if not torch.allclose(gram, gram.T):
        error_msg = "Non-symmetric Gram matrix detected"
        print(f"[CKA ERROR] {error_msg}")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg})
        raise ValueError(error_msg)
        
    G = gram.clone()
    N = G.size(0)

    if unbiased:
        # Check batch size for unbiased version
        if N <= 2:
            error_msg = f"Batch size {N} too small for unbiased centering. Need at least 3 samples."
            print(f"[CKA ERROR] {error_msg}")
            if torch.distributed.get_rank() == 0:
                wandb.log({"error": error_msg})
            raise ValueError(error_msg)
            
        # U‐centered version
        G.fill_diagonal_(0)
        # row / col means
        means = G.sum(dim=0) / (N - 2)
        means = means - means.sum() / (2 * (N - 1))
        G = G - means[None] - means[:, None]
        G.diagonal().zero_()
    else:
        # ordinary centering
        row_means = G.mean(dim=0, keepdim=True)
        col_means = G.mean(dim=1, keepdim=True)
        total_mean = row_means.mean()
        G = G - row_means - col_means + total_mean

    if torch.isnan(G).any():
        error_msg = f"NaN detected in centered Gram matrix. Matrix stats: min={G.min().item() if not torch.isnan(G.min()) else 'NaN'}, max={G.max().item() if not torch.isnan(G.max()) else 'NaN'}, shape={G.shape}"
        print(f"[CKA ERROR] {error_msg}")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg})
        raise ValueError(error_msg)
        
    return G

def cka(gram_x: torch.Tensor, gram_y: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """
    Compute linear CKA between two Gram matrices.
    Returns a single scalar tensor.
    """
    Gx = center_gram(gram_x, unbiased)
    Gy = center_gram(gram_y, unbiased)

    # HSIC numerator
    hsic = (Gx * Gy).sum()
    # normalizers
    norm_x = Gx.norm()
    norm_y = Gy.norm()
    
    # Check for tiny norms that could cause numerical instability
    epsilon = 1e-10
    if norm_x < epsilon or norm_y < epsilon:
        if torch.allclose(gram_x[0], gram_x[1:], rtol=1e-5, atol=1e-5):
            print("Identical text tokens found in sample")
        error_msg = f"Near-zero norm detected: norm_x={norm_x.item()}, norm_y={norm_y.item()}. This will cause NaN in CKA."
        input_tensors_msg = f"Input tensors: gram_x ={gram_x}, gram_y ={gram_y}"
        centered_tensors_msg = f"Centered tensors: Gx ={Gx}, Gy ={Gy}"
        print(f"[CKA ERROR] {error_msg}")
        print(f"[CKA DEBUG] {input_tensors_msg}")
        print(f"[CKA DEBUG] {centered_tensors_msg}")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg, "input_tensors": input_tensors_msg, "centered_tensors": centered_tensors_msg})
        raise ValueError(error_msg)
    
    result = hsic / (norm_x * norm_y)
    
    if torch.isnan(result):
        error_msg = f"NaN in CKA result. hsic={hsic.item()}, norm_x={norm_x.item()}, norm_y={norm_y.item()}"
        print(f"[CKA ERROR] {error_msg}")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg})
        raise ValueError(error_msg)
        
    return result

def compute_cka_unbiased(text_embeds: torch.Tensor, image_embeds: torch.Tensor, unbiased: bool = False, model: torch.nn.Module = None) -> torch.Tensor:
    """
    text_embeds: [B, D_text] feature matrices (B = batch size, D_text = number of text tokens * text embedding dim)
    image_embeds: [B, D_img] feature matrices (B = batch size, D_img = number of image tokens * image embedding dim)
    returns: scalar CKA similarity
    Xflat  # [batch_size, N_img * hidden_dim]
    Yflat  # [batch_size, max(N_txt) * hidden_dim]
    """
    if torch.allclose(text_embeds[0], text_embeds[1:], rtol=1e-5, atol=1e-5):
        print("Identical text tokens found in text_embeds")
        print(f"text_embeds: {text_embeds}")
    # Check for NaN in input embeddings
    if torch.isnan(text_embeds).any():
        error_msg = f"NaN detected in text_embeds. Shape: {text_embeds.shape}, stats: min={text_embeds.min().item() if not torch.isnan(text_embeds.min()) else 'NaN'}, max={text_embeds.max().item() if not torch.isnan(text_embeds.max()) else 'NaN'}"
        print(f"[CKA ERROR] {error_msg}")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg})
        raise ValueError(error_msg)
    if torch.isnan(image_embeds).any():
        error_msg = f"NaN detected in image_embeds. Shape: {image_embeds.shape}, stats: min={image_embeds.min().item() if not torch.isnan(image_embeds.min()) else 'NaN'}, max={image_embeds.max().item() if not torch.isnan(image_embeds.max()) else 'NaN'}"
        print(f"[CKA ERROR] {error_msg}")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg})
        raise ValueError(error_msg)
    
    # Check for zero variance
    text_var = text_embeds.var(dim=1)
    image_var = image_embeds.var(dim=1)
    if text_var.min() < 1e-10:
        error_msg = f"Near-zero variance in text_embeds. Variance stats: min={text_var.min().item()}, max={text_var.max().item()}"
        print(f"[CKA ERROR] {error_msg}")
        print(f"[CKA DEBUG] text_embeds shape: {text_embeds.shape}")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg})
        raise ValueError(error_msg)
    if image_var.min() < 1e-10:
        error_msg = f"Near-zero variance in image_embeds. Variance stats: min={image_var.min().item()}, max={image_var.max().item()}"
        print(f"[CKA ERROR] {error_msg}")
        print(f"[CKA DEBUG] image_embeds shape: {image_embeds.shape}")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg})
        raise ValueError(error_msg)
        
    Gx = gram_linear(text_embeds)
    Gy = gram_linear(image_embeds)
    if torch.allclose(Gx[0], Gx[1:], rtol=1e-5, atol=1e-5):
        print("Identical text tokens found in gram_linear after matrix multiplication")
        print(f"Gx: {Gx}")
        print(f"Text embeds: {text_embeds}")
        print("==========================\n")
        are_identical, _, decoded_text = check_and_reconstruct_identical_tokens(
            text_embeds,
            tokenizer=AutoTokenizer.from_pretrained(
                model.config._name_or_path,
                use_fast=False,
                padding_side="right"
            ),
            model=model,
            print_info=True
        )
        print(f"Decoded text: {decoded_text}")
        print("==========================\n")
    Gx_centered = center_gram(Gx, unbiased)
    Gy_centered = center_gram(Gy, unbiased)

    # HSIC numerator
    # normalizers
    norm_x = Gx_centered.norm()
    norm_y = Gy_centered.norm()
    
    # Check for tiny norms that could cause numerical instability
    epsilon = 1e-10
    if norm_x < epsilon or norm_y < epsilon:
        if torch.allclose(Gx_centered[0], Gx_centered[1:], rtol=1e-5, atol=1e-5):
            print("Identical text tokens found in sample")
        error_msg = f"Near-zero norm detected: norm_x={norm_x.item()}, norm_y={norm_y.item()}. This will cause NaN in CKA."
        input_tensors_msg = f"Input tensors: gram_x ={Gx}, gram_y ={Gy}"
        centered_tensors_msg = f"Centered tensors: Gx ={Gx_centered}, Gy ={Gy_centered}"
        print(f"[CKA ERROR] {error_msg}")
        print(f"[CKA DEBUG] {input_tensors_msg}")
        print(f"[CKA DEBUG] {centered_tensors_msg}")
        print(f"Text embeds: {text_embeds}")
        print("==========================\n")
        are_identical, _, decoded_text = check_and_reconstruct_identical_tokens(
            text_embeds,
            tokenizer=AutoTokenizer.from_pretrained(
                model.config._name_or_path,
                use_fast=False,
                padding_side="right"
            ),
            model=model,
            print_info=True
        )
        print(f"Decoded text: {decoded_text}")
        print("==========================\n")
        if torch.distributed.get_rank() == 0:
            wandb.log({"error": error_msg, "input_tensors": input_tensors_msg, "centered_tensors": centered_tensors_msg})
        raise ValueError(error_msg)
      

    try:
        result = cka(Gx, Gy, unbiased).item()
        if math.isnan(result):
            error_msg = f"NaN in final CKA result after conversion to Python float"
            print(f"[CKA ERROR] {error_msg}")
            if torch.distributed.get_rank() == 0:
                wandb.log({"error": error_msg})
            raise ValueError(error_msg)
        return result
    except Exception as e:
        error_msg = f"Exception in CKA computation: {e}"
        debug_info = f"Text embeds shape: {text_embeds.shape}, Image embeds shape: {image_embeds.shape}, Gram matrix shapes: Gx={Gx.shape}, Gy={Gy.shape}"
        print(f"[CKA ERROR] {error_msg}")
        print(f"[CKA DEBUG] {debug_info}")
        if torch.distributed.get_rank() == 0:
            wandb.log({
                "error": error_msg,
                "debug/text_embeds_shape": str(text_embeds.shape),
                "debug/image_embeds_shape": str(image_embeds.shape),
                "debug/Gx_shape": str(Gx.shape),
                "debug/Gy_shape": str(Gy.shape)
            })
        raise

def compute_cosine_similarity(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute per-sample cosine similarity between two sets of feature vectors.
    Args:
      X: Tensor of shape [B, D_text]
      Y: Tensor of shape [B, D_img]
      eps: small constant to avoid divide-by-zero
    Returns:
      cos_sims: Tensor of shape [B], where
        cos_sims[i] = (X[i] · Y[i]) / (||X[i]|| * ||Y[i]||)
    """
    # F.cosine_similarity does exactly this under the hood
    x_mean = X.mean(dim=0)
    y_mean = Y.mean(dim=0)
    cos_sims = F.cosine_similarity(x_mean, y_mean, dim=0, eps=eps)
    return cos_sims



def compute_mmd_linear(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the **linear** Maximum Mean Discrepancy (MMD) between two batches.
    Computes per-sample MMD by comparing the mean of each text and image embedding sequence.

    Args:
        X: Tensor of shape [B, N_text, D]  - text embeddings
        Y: Tensor of shape [B, N_img, D]   - image embeddings

    Returns:
        Scalar tensor: the mean of per-sample MMDs
    """
    # print(f"X: {X}")
    # print(f"Y: {Y}")
    if X.size(1) != Y.size(1):
        raise ValueError("Feature dims differ.")
    mu_t = X.mean(dim=0)     # [D]
    mu_i = Y.mean(dim=0)    # [D]
    # print(f"mu_t: {mu_t}")
    # print(f"mu_i: {mu_i}")
    return (mu_t - mu_i).pow(2).mean()  # scalar