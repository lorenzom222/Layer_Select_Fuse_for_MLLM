from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast
import torch.nn.functional as F

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


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

# Function to compute the Hilbert-Schmidt Independence Criterion
def hsic(K, L):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC).
    Args:
        K (torch.Tensor): First Gram matrix (n x n).
        L (torch.Tensor): Second Gram matrix (n x n).
    Returns:
        torch.Tensor: The HSIC value.
    """
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - 1.0 / n * torch.ones(n, n, device=K.device)
    K_c = torch.matmul(torch.matmul(H, K), H)
    L_c = torch.matmul(torch.matmul(H, L), H)
    # The Frobenius norm squared is equivalent to the trace of K_c.T @ L_c
    # For symmetric matrices K_c, L_c, this is trace(K_c @ L_c)
    criterion = torch.trace(torch.matmul(K_c, L_c))
    return criterion / ((n - 1) ** 2) # Unbiased estimator factor

def align_embeddings_for_cka(features_x: torch.Tensor, features_y: torch.Tensor, pooling: str = 'mean', projection_size: int = 64):
    """
    Aligns two feature matrices by applying pooling if their number of samples (rows) differ.

    Args:
        features_x (torch.Tensor): First feature matrix of shape [N_x, D]
        features_y (torch.Tensor): Second feature matrix of shape [N_y, D]
        pooling (str): Pooling strategy to apply if N_x ≠ N_y. Options: 'mean', 'max', or 'none'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Aligned feature matrices of shape [N, D] or [1, D]
    """
    features_x = features_x.float()
    features_y = features_y.float()

    if features_x.dim() > 2:
        features_x = features_x.view(features_x.shape[0], -1)
    if features_y.dim() > 2:
        features_y = features_y.view(features_y.shape[0], -1)

    if features_x.shape[0] != features_y.shape[0]:
        if pooling == 'mean':
            features_x = features_x.mean(dim=0, keepdim=True)
            features_y = features_y.mean(dim=0, keepdim=True)
            # print("Features X shape after mean pooling:", features_x.shape)
            # print("Features Y shape after mean pooling:", features_y.shape)
        elif pooling == 'max':
            features_x = features_x.max(dim=0, keepdim=True).values
            features_y = features_y.max(dim=0, keepdim=True).values
            # print("Features X shape after max pooling:", features_x.shape)
            # print("Features Y shape after max pooling:", features_y.shape)
        elif pooling == 'none':
            raise ValueError(
                f"CKA input mismatch: features_x has {features_x.shape[0]} samples, "
                f"features_y has {features_y.shape[0]} — set pooling='mean' or 'max' to resolve."
            )
        elif pooling == 'interpolate':
            # Resize to fixed length using linear interpolation (1D)
            def resize(tensor, N):
                tensor = tensor.unsqueeze(0).transpose(1, 2)  # [1, D, L]
                tensor = F.interpolate(tensor, size=N, mode='linear', align_corners=False)
                return tensor.squeeze(0).transpose(0, 1)  # [N, D]

            features_x = resize(features_x, projection_size)
            features_y = resize(features_y, projection_size)
            # print(f"Interpolated to shape: {features_x.shape}")
        else:
            raise ValueError(f"Invalid pooling type: '{pooling}'. Choose 'mean', 'max', or 'none'.")

    return features_x, features_y


# Function to compute the Centered Kernel Alignment (CKA) similarity (unbiased)
def unbiased_cka(features_x, features_y, pooling: str = 'interpolate'):
    """
    Computes the unbiased Centered Kernel Alignment (CKA) similarity between two feature matrices.
    If the number of samples (rows) does not match, applies pooling (mean or max) to reduce each to a single vector.

    Args:
        features_x (torch.Tensor): First feature matrix (n_samples_x, n_features).
        features_y (torch.Tensor): Second feature matrix (n_samples_y, n_features).
        pooling (str): Pooling strategy if n_samples mismatch. Options: 'mean' (default), 'max', 'none'.

    Returns:
        torch.Tensor: The CKA similarity value.
    """
    # print("Features X shape before alignment:", features_x.shape)
    # print("Features Y shape before alignment:", features_y.shape)
    features_x, features_y = align_embeddings_for_cka(features_x, features_y, pooling=pooling)



    # Compute Gram matrices
    gram_x = torch.matmul(features_x, features_x.t())
    gram_y = torch.matmul(features_y, features_y.t())

    # Compute HSIC
    hsic_xy = hsic(gram_x, gram_y)
    hsic_xx = hsic(gram_x, gram_x)
    hsic_yy = hsic(gram_y, gram_y)

    # CKA similarity
    epsilon = 1e-5
    cka_value = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy + epsilon))
    if torch.isnan(cka_value):
        print("⚠️ Warning: CKA returned NaN")
        print("HSIC XY:", hsic_xy.item())
        print("HSIC XX:", hsic_xx.item())
        print("HSIC YY:", hsic_yy.item())

    return cka_value