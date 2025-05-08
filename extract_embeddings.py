# TODO: Need to parallelize this across our 8 GPUs. Rn just works on CPU. Not good.
# NOTE: Maybe check out how accelerate works.
import argparse
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoModel
from tqdm import tqdm
import logging
from typing import Dict
import torch.nn.functional as F # Needed for padding
import wandb
from accelerate import Accelerator # Add Accelerator
# from accelerate.utils import gather_object # May not be needed if gathering tensors

# --- CKA Helper Functions (Adapted from NumPy version) ---

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features tensor of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    return torch.matmul(x, x.T)

def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric tensor.
      unbiased: Whether to adjust the Gram matrix to compute an unbiased
        estimate of HSIC.

    Returns:
      A symmetric tensor with centered columns and rows.
    """
    if not torch.allclose(gram, gram.T):
        # Note: Floating point inaccuracies can sometimes lead to slight asymmetry.
        # Consider adding a tolerance check if needed: torch.allclose(gram, gram.T, atol=1e-6)
        # For now, we raise an error for clear non-symmetry.
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.clone()
    n = gram.shape[0]

    if unbiased:
        # This formulation of the U-statistic seems more numerically stable.
        # Equivalent to: H*gram*H where H = I - 1/n * 11^T - diag(1/n ... 1/n)
        # But uses adjusted means for stability.
        gram.fill_diagonal_(0)
        means = torch.sum(gram, dim=0) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)
    else:
        # Equivalent to: H*gram*H where H = I - 1/n * 11^T
        means = torch.mean(gram, dim=0)
        means -= torch.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram

def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # HSIC = trace(centered_gram_x @ centered_gram_y)
    # scaled_hsic = ||centered_gram_x @ centered_gram_y||_F^2 ??? No, dot product is correct.
    # The HCSIC computation simplifies for CKA calculation.
    scaled_hsic = torch.dot(gram_x.reshape(-1), gram_y.reshape(-1))

    normalization_x = torch.linalg.norm(gram_x, ord='fro')
    normalization_y = torch.linalg.norm(gram_y, ord='fro')
    
    # Handle potential division by zero
    if normalization_x == 0 or normalization_y == 0:
        return torch.tensor(0.0, device=gram_x.device) 
    
    return scaled_hsic / (normalization_x * normalization_y)

# --- End CKA Helper Functions ---

# --- CKA Computation Helper ---
def compute_and_print_cka(vit_embeddings_device, llm_hidden_states_device, batch_idx, args):
    """Computes and logs CKA between ViT embeddings and each LLM layer's hidden states.

    Args:
        vit_embeddings_device: Tensor of ViT embeddings on the compute device.
        llm_hidden_states_device: List of Tensors of LLM hidden states on the compute device.
        batch_idx: The current batch index (for logging).
        args: Command-line arguments/configuration (for verbose_logging flag).
    """
    if vit_embeddings_device is None or llm_hidden_states_device is None:
        logging.warning(f"[Batch {batch_idx}] Skipping CKA due to missing ViT or LLM embeddings.")
        return

    if args.verbose_logging:
        logging.info(f"--- [Batch {batch_idx}] CKA Results (ViT vs LLM Layers) ---")
    
    I = vit_embeddings_device # Shape [N, Ti, Di]
    N, Ti, Di = I.shape
    
    for layer_idx, T in enumerate(llm_hidden_states_device):
        # T shape is [N, Tt, Dt]
        _, Tt, Dt = T.shape
        
        # Ensure N matches (batch size consistency)
        if I.shape[0] != T.shape[0]:
            logging.warning(f"[Batch {batch_idx}, Layer {layer_idx}] Batch size mismatch between ViT ({I.shape[0]}) and LLM ({T.shape[0]}). Skipping CKA.")
            continue

        cka_log_prefix = f"[Batch {batch_idx}, Layer {layer_idx}] CKA"
        # Method 1: Flattened CKA
        try:
            X_flat = I.reshape(N, -1) # [N, Ti * Di]
            Y_flat = T.reshape(N, -1) # [N, Tt * Dt]
            
            gram_x_flat = gram_linear(X_flat)
            gram_y_flat = gram_linear(Y_flat)
            
            cka_flattened_val = cka(gram_x_flat, gram_y_flat, debiased=True)
            if args.verbose_logging:
                logging.info(f"{cka_log_prefix} (Flattened) = {cka_flattened_val:.4f}")
            # Log to wandb (always log to wandb)
            wandb.log({f"CKA/Layer_{layer_idx}/Flattened": cka_flattened_val.item(), "batch": batch_idx})
        except Exception as e:
            # Always log errors
            logging.error(f"{cka_log_prefix} (Flattened) = ERROR ({e})")

        # Method 2: Padded CKA
        try:
            T_max = max(Ti, Tt)
            
            # Pad I (images) if Ti < T_max
            if Ti < T_max:
                padding_i = (0, 0, 0, T_max - Ti) # Pad last dim (tokens), then features
                Ipad = F.pad(I, padding_i, "constant", 0)
            else:
                Ipad = I
            
            # Pad T (text) if Tt < T_max
            if Tt < T_max:
                padding_t = (0, 0, 0, T_max - Tt) # Pad last dim (tokens), then features
                Tpad = F.pad(T, padding_t, "constant", 0)
            else:
                Tpad = T
            
            X_pad = Ipad.reshape(N, -1) # [N, T_max * Di]
            Y_pad = Tpad.reshape(N, -1) # [N, T_max * Dt]
            
            gram_x_pad = gram_linear(X_pad)
            gram_y_pad = gram_linear(Y_pad)
            
            cka_padded_val = cka(gram_x_pad, gram_y_pad, debiased=True)
            if args.verbose_logging:
                logging.info(f"{cka_log_prefix} (Padded)   = {cka_padded_val:.4f}")
            # Log to wandb (always log to wandb)
            wandb.log({f"CKA/Layer_{layer_idx}/Padded": cka_padded_val.item(), "batch": batch_idx})
        except Exception as e:
            # Always log errors
            logging.error(f"{cka_log_prefix} (Padded)   = ERROR ({e})")
            
    if args.verbose_logging:
        logging.info(f"--- [Batch {batch_idx}] End CKA Results ---")

# --- End CKA Computation Helper ---

# --- Embedding Extraction Helper ---
def extract_embeddings_for_batch(batch, vit_model, llm_model, args, torch_dtype, batch_idx):
    """Performs forward passes and extracts embeddings for a single batch.

    Args:
        batch: Dictionary containing batch data (pixel_values, input_ids, etc.).
        vit_model: The Vision Transformer model.
        llm_model: The Language Model.
        args: Command-line arguments/configuration.
        torch_dtype: The torch data type to use.
        batch_idx: Current batch index (for logging).

    Returns:
        A tuple (vit_embeddings_device, llm_hidden_states_device).
        Returns (None, None) if extraction fails for either modality.
    """
    # Move batch to device (redundant if already done, but safe)
    try:
        pixel_values = batch['pixel_values'].to(args.device, dtype=torch_dtype) if batch['pixel_values'] is not None else None
        input_ids = batch['input_ids'].to(args.device) if batch['input_ids'] is not None else None
        attention_mask = batch['attention_mask'].to(args.device) if batch['attention_mask'] is not None else None
    except Exception as e:
        logging.error(f"Error moving batch {batch_idx} data to device {args.device} within extraction: {e}")
        return None, None

    # --- ViT Forward Pass ---
    vit_embeddings_device = None
    if pixel_values is not None:
        try:
            vit_outputs = vit_model(pixel_values=pixel_values)
            vit_embeddings_device = vit_outputs.last_hidden_state # Keep on device
        except Exception as e:
            logging.error(f"Error during ViT forward pass for batch {batch_idx}: {e}")
            # If ViT fails, we might still want LLM, but let's return None for simplicity now
            # return None, None # Option 1: Fail whole batch extraction
            vit_embeddings_device = None # Option 2: Continue and return None for ViT
    else:
        logging.warning(f"No valid images found in batch {batch_idx}. Skipping ViT forward pass.")

    # --- LLM Forward Pass ---
    llm_hidden_states_device = None
    if input_ids is not None and attention_mask is not None:
        try:
            llm_outputs = llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            llm_hidden_states_device = llm_outputs.hidden_states # Keep on device
        except Exception as e:
            logging.error(f"Error during LLM forward pass for batch {batch_idx}: {e}")
            # If LLM fails, return None for LLM
            llm_hidden_states_device = None
    else:
         logging.warning(f"No valid text input found in batch {batch_idx}. Skipping LLM forward pass.")

    return vit_embeddings_device, llm_hidden_states_device

# --- End Embedding Extraction Helper ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataset Definition ---
class ImageTextDataset(Dataset):
    def __init__(self, data_path: str, image_folder: str):
        logging.info(f"Loading dataset index from: {data_path}")
        try:
            self.list_data_dict = json.load(open(data_path, "r"))
            # Basic validation of the first item
            if self.list_data_dict and isinstance(self.list_data_dict[0], dict):
                item = self.list_data_dict[0]
                if 'image' not in item or 'conversations' not in item:
                     raise ValueError("Dataset items must contain 'image' and 'conversations' keys.")
            elif not self.list_data_dict:
                 raise ValueError("Dataset JSON is empty.")
            else:
                raise ValueError("Dataset JSON format is not a list of dictionaries.")
            logging.info(f"Loaded {len(self.list_data_dict)} items from dataset index.")
        except Exception as e:
            logging.error(f"Error loading or parsing dataset index: {e}")
            raise
        self.image_folder = image_folder

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, any]:
        item = self.list_data_dict[i]
        image_file = item.get('image')
        conversations = item.get('conversations', [])

        if not image_file:
            logging.warning(f"Missing 'image' key in item {i}. Skipping image.")
            image_file = None # Or handle differently

        # Concatenate conversation values to form the text
        text = "".join([conv.get('value', '') for conv in conversations])
        if not text:
             logging.warning(f"Empty text generated for item {i}.")

        return {
            "id": item.get('id', f'item_{i}'), # Use provided ID or generate one
            "image_file": image_file,
            "text": text.strip()
        }

# --- Collate Function ---
def create_collate_fn(tokenizer, image_processor, image_folder, device, model_max_length=512):
    def collate_fn(batch):
        image_files = [item['image_file'] for item in batch if item['image_file'] is not None]
        texts = [item['text'] for item in batch]
        ids = [item['id'] for item in batch]

        pixel_values = None
        if image_files:
            images = []
            valid_image_indices = [] # Keep track of which items in the batch have valid images
            for i, img_file in enumerate(image_files):
                 try:
                    img_path = os.path.join(image_folder, img_file)
                    images.append(Image.open(img_path).convert('RGB'))
                    valid_image_indices.append(i) # Assuming image loading/processing works for this index
                 except Exception as e:
                    logging.warning(f"Could not load image {img_file}: {e}. Skipping image for this item.")
            
            if images:
                try:
                    processed_images = image_processor(images=images, return_tensors='pt')
                    pixel_values = processed_images['pixel_values']
                except Exception as e:
                    logging.error(f"Error processing images with image_processor: {e}")
                    # Decide how to handle batch if image processing fails (e.g., return None, skip batch)
                    pixel_values = None # Or handle error differently

        try:
            tokenized_text = tokenizer(
                texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=model_max_length # Use tokenizer's max length or a specific value
            )
        except Exception as e:
            logging.error(f"Error tokenizing texts: {e}")
            # Handle tokenization error (e.g., return None for text fields)
            tokenized_text = {'input_ids': None, 'attention_mask': None}

        return {
            'ids': ids,
            'image_files': image_files, # List of image files corresponding to pixel_values
            'pixel_values': pixel_values, # Tensor or None
            'input_ids': tokenized_text['input_ids'], # Tensor or None
            'attention_mask': tokenized_text['attention_mask'] # Tensor or None
        }
    return collate_fn

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from Vision Transformer and Language Model.")
    parser.add_argument("--model_name_or_path", type=str, 
                        default="mtgv/MobileLLaMA-1.4B-Base", 
                        help="Path or name of the pretrained language model.")
    parser.add_argument("--vision_tower", type=str, 
                        default="google/siglip-so400m-patch14-384", 
                        help="Path or name of the pretrained vision tower (e.g., CLIP).")
    parser.add_argument("--data_path", type=str, 
                        default="./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json", 
                        help="Path to the dataset JSON file.")
    parser.add_argument("--image_folder", type=str, 
                        default="./playground/data/LLaVA-Pretrain/images", 
                        help="Path to the folder containing images.")
    parser.add_argument("--output_dir", type=str, 
                        default="./extracted_embeddings", 
                        help="Directory to save the extracted embeddings.")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for processing.")
    parser.add_argument("--model_max_length", type=int, default=2048, 
                        help="Maximum sequence length for the LLM tokenizer.")
    parser.add_argument("--max_batches", type=int, default=None, 
                        help="Maximum number of batches to process (optional).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu).")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        choices=["float16", "bfloat16", "float32"], 
                        help="Data type for models (e.g., float16, bfloat16, float32).")
    parser.add_argument("--verbose_logging", action='store_true', 
                        help="Enable detailed logging during batch processing (CKA scores, shapes).")
    # Pass an empty list to parse_args to avoid conflicts with kernel arguments
    args = parser.parse_args([]) 
    # Ensure output_dir is provided - Removing this check as default is set
    # if not args.output_dir:
    #     parser.error("--output_dir is required.")
    return args

# --- Helper Functions ---
def load_models(args):
    logging.info("Loading models and tokenizer...")
    # Determine torch dtype
    torch_dtype = torch.float32
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    logging.info(f"Using torch dtype: {torch_dtype}")

    # Load LLM Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")

    # Load LLM Model
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype
    ).to(args.device).eval()

    # Load ViT Processor
    image_processor = AutoImageProcessor.from_pretrained(args.vision_tower)

    # Load ViT Model
    vit_model = AutoModel.from_pretrained(
        args.vision_tower,
        torch_dtype=torch_dtype
    ).to(args.device).eval()
    if hasattr(vit_model, 'vision_model'):
        vit_model = vit_model.vision_model

    logging.info("Models loaded.")
    return tokenizer, llm_model, image_processor, vit_model, torch_dtype

def load_data(args, tokenizer, image_processor):
    logging.info("Loading dataset...")
    dataset = ImageTextDataset(data_path=args.data_path, image_folder=args.image_folder)
    collate_fn = create_collate_fn(
        tokenizer,
        image_processor,
        args.image_folder,
        args.device,
        model_max_length=args.model_max_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4 # Adjust as needed
    )
    logging.info("Dataset loaded.")
    return dataloader

def process_batches(args, dataloader, llm_model, vit_model, torch_dtype):
    logging.info("Starting processing loop...")
    batch_count = 0
    total_items_processed = 0
    # Prepare subdirectories for outputs
    vit_output_dir = os.path.join(args.output_dir, "vit_embeddings")
    llm_output_dir = os.path.join(args.output_dir, "llm_hidden_states")
    os.makedirs(vit_output_dir, exist_ok=True)
    os.makedirs(llm_output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                logging.info(f"Reached max_batches limit ({args.max_batches}). Stopping.")
                break

            # Move batch to device
            try:
                # We still need batch_ids here
                batch_ids = batch['ids']
                # Moving tensors to device is now handled within extract_embeddings_for_batch
            except Exception as e:
                logging.error(f"Error getting batch IDs for batch {batch_idx}: {e}")
                continue # Skip this batch
            
            # --- Embedding Extraction --- 
            vit_embeddings, llm_hidden_states_device = extract_embeddings_for_batch(
                batch, vit_model, llm_model, args, torch_dtype, batch_idx
            )

            # --- CKA Computation --- 
            compute_and_print_cka(vit_embeddings, llm_hidden_states_device, batch_idx, args)

            # --- Process Outputs (Log Shapes) ---
            if args.verbose_logging:
                logging.info(f"--- [Batch {batch_idx}] Artifact Shapes --- (Embeddings kept on {args.device}) ---")
                if vit_embeddings is not None:
                    logging.info(f"  [Batch {batch_idx}] ViT Embeddings Shape: {vit_embeddings.shape}")
                    # logging.info(f"  [Batch {batch_idx}] ViT Embeddings (Sample):\n{vit_embeddings[0, 0, :5]}...")
                else:
                    # Warning is already logged if missing, this info log isn't strictly needed but ok
                    logging.info(f"  [Batch {batch_idx}] ViT Embeddings: Not computed or error occurred.")

                if llm_hidden_states_device is not None:
                    logging.info(f"  [Batch {batch_idx}] LLM Hidden States: {len(llm_hidden_states_device)} layers")
                    logging.info(f"    [Batch {batch_idx}] Layer 0 (Embeddings) Shape: {llm_hidden_states_device[0].shape}")
                    logging.info(f"    [Batch {batch_idx}] Layer {len(llm_hidden_states_device)-1} (Final) Shape: {llm_hidden_states_device[-1].shape}")
                    # logging.info(f"    [Batch {batch_idx}] LLM Layer 0 (Sample):\n{llm_hidden_states_device[0][0, 0, :5]}...")
                    # logging.info(f"    [Batch {batch_idx}] LLM Layer {len(llm_hidden_states_device)-1} (Sample):\n{llm_hidden_states_device[-1][0, 0, :5]}...")
                else:
                    # Warning is already logged if missing, this info log isn't strictly needed but ok
                    logging.info(f"  [Batch {batch_idx}] LLM Hidden States: Not computed or error occurred.")
                logging.info(f"--- [Batch {batch_idx}] End Artifact Shapes ---")

            batch_count += 1
            total_items_processed += len(batch_ids)

    logging.info(f"Processing finished. Processed {batch_count} batches ({total_items_processed} items). Outputs saved to {args.output_dir}")

def main():
    args = parse_args()
    logging.info(f"Starting embedding extraction with args: {args}")

    # <<<--- Add this line for testing on a small sample --->>>
    # args.max_batches = 1 # Process only the first batch for testing
    # logging.warning(f"*** TEST MODE: Limiting processing to {args.max_batches} batch(es) ***")
    # <<<--------------------------------------------------->>>

    # --- Initialize Wandb ---
    try:
        run_name = f"CKA_Extract_{os.path.basename(args.model_name_or_path)}_{os.path.basename(args.vision_tower)}"
        wandb.init(
            project="cka-embedding-extraction", # Or your preferred project name
            name=run_name,
            config=vars(args) # Log arguments
        )
        logging.info(f"Wandb initialized for run: {run_name}")
    except Exception as e:
        logging.error(f"Failed to initialize wandb: {e}. Proceeding without wandb logging.")
        wandb.init(mode="disabled") # Disable wandb if init fails
    # ------------------------

    # Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Models and Tokenizer ---
    tokenizer, llm_model, image_processor, vit_model, torch_dtype = load_models(args)

    # --- Load Data ---
    dataloader = load_data(args, tokenizer, image_processor)

    # --- Process Batches --- 
    process_batches(args, dataloader, llm_model, vit_model, torch_dtype)

    logging.info("Embedding extraction complete.")
    wandb.finish() # Ensure wandb run finishes

if __name__ == "__main__":
    main() 