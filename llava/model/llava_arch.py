#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        
        # init vision tower
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)

            # init fusion strat (eg. Internal Direct, External Direct, Internal Modular, External Modular)
            # here we just use Internal Direct/Modular
            if "I" in self.config.layer_fusing_strategy:
                # then we pick vit layer selection strat
                if self.config.layer_using_strategy == 'former' or self.config.layer_using_strategy == 'latter':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(12)])
                elif self.config.layer_using_strategy == 'all':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(24)])
                elif self.config.layer_using_strategy == '3-18-23':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(3)])
                elif self.config.layer_using_strategy == '3-18':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(2)])
                elif self.config.layer_using_strategy == '18':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(1)])
                elif self.config.layer_using_strategy == 'last':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(1)])

            self.mm_projector_f = build_vision_projector(config, vision_tower=self.vision_tower) # this is for ... not sure??? building the projector for the fusion

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''): 
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


    def initialize_vision_modules(self, model_args, fsdp=None):
        '''
        This function initializes all the vision-related components of the LLaVA model.
        
        In simple terms, it:
        1. Sets up the vision encoder (like SigLIP or CLIP) that processes images
        2. Configures how the model selects which layers from the vision encoder to use
        3. Creates the projectors that transform visual features into a format the language model can understand
        4. Loads pretrained weights for these projectors if available
        
        Think of it as the "setup crew" that prepares all the vision components before the model starts working.
        It's like assembling all the parts of a camera before you can take pictures.
        
        The function takes model arguments (settings) and optionally fsdp (for distributed training).
        From the arg: --pretrain_mm_mlp_adapter ./checkpoint/${BASE_MODEL_NAME}-${FUSING_STRATEGY}-pretrain-${USING_STRATEGY}-${MODEL_NAME}/mm_projector.bin
            What is the mm_projector? 
        --mm_vision_select_layer -2
        --mm_use_im_start_end False
        --mm_use_im_patch_token False
        --image_aspect_ratio pad
        '''
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        # Vision tower setup
        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        # Config updates
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.n_queries = getattr(model_args, 'n_queries', None)
        self.config.layer_using_strategy = getattr(model_args, 'layer_using_strategy', None)
        self.config.layer_fusing_strategy = getattr(model_args, 'layer_fusing_strategy', None)
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        # Projector initialization
        if getattr(self, 'mm_projectors', None) is None:
            # Path 1: Initialize New Projectors
            if "I" in self.config.layer_fusing_strategy:
                # if we are using internal fusion, then we need to initialize the projectors
                if self.config.layer_using_strategy == 'former' or self.config.layer_using_strategy == 'latter':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(12)])
                elif self.config.layer_using_strategy == 'all':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(24)])
                elif self.config.layer_using_strategy == '3-18-23':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(3)])
                elif self.config.layer_using_strategy == '3-18':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(2)])
                elif self.config.layer_using_strategy == '18':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(1)])
                elif self.config.layer_using_strategy == 'last':
                    self.mm_projectors = nn.ModuleList([build_vision_projector(self.config, vision_tower=self.vision_tower) for _ in range(1)])
            self.mm_projector_f = build_vision_projector(self.config, vision_tower=self.vision_tower)
            if 'unpad' in mm_patch_merge_type:
                # if we are using unpad, then we need to initialize the image_newline
                # what is unpad? I think it's for...
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # Path 2: Using Existing Projectors
            # why do we need to enable grad for this but not for if not initialized? bc loading the pretrained projectors sets requires_grad to False. we want to set it to True bc we still want to train them for finetuning.
            for idx, projector in enumerate(self.mm_projectors):
                for name, param in projector.named_parameters():
                    param.requires_grad = True
                    print(f"Enabled grad for mm_projectors[{idx}]: {name}")
            for name, param in self.mm_projector_f.named_parameters():
                param.requires_grad = True
                print(f"Enabled grad for mm_projector_f: {name}")

        # Load pretrained weights for projectors
        if pretrain_mm_mlp_adapter is not None:
            # helper function to extract weights from a dictionary
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            if "I" in self.config.layer_fusing_strategy:
                # for Internal Fusion, loads pretrained weights for each projector
                for idx in range(len(self.mm_projectors)):
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter.replace('mm_projector.bin', f'mm_projector_{idx}.bin'), map_location='cpu')
                    print(idx)
                    a = get_w(mm_projector_weights, f'mm_projectors.{idx}')
                    self.mm_projectors[idx].load_state_dict(a)
                    print(f'mm_projector_{idx} is loaded!!!')

            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter.replace('mm_projector.bin', 'mm_projector_f.bin'), map_location='cpu')
            self.mm_projector_f.load_state_dict(get_w(mm_projector_weights, 'mm_projector_f'))
            print('mm_projector_f is loaded!!!')

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    '''
    Abstract base class for LLaVA models that perform causal language modeling (predicting next token in a sequence)
    How does it link to the LlavaMetaModel class?
    
    Causal vs non-causal:
    - Causal: predict next token in a sequence
    - Non-causal: predict any token in the sequence, including the next one
    
    Core part of the class:
    - get_model(): returns the model
    - get_vision_tower(): returns the vision tower
    - encode_images(): encodes images
    - prepare_inputs_labels_for_multimodal(): prepares inputs and labels for multimodal training
    - initialize_vision_modules(): initializes the vision modules
    '''

    @abstractmethod
    def get_model(self):
        '''
        To be implemented elsewhere. EX:

        > class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
        >    def __init__(self, config):
        >        super().__init__(config)
        >        self.model = LlavaLlamaModel(config)  # Create the LlavaMetaModel instance
        >        
        >    def get_model(self):
        >        return self.model  # Return the LlavaMetaModel instance
        '''
        pass

    def get_vision_tower(self):
        '''
        Getter method for the vision tower
        '''
        return self.get_model().get_vision_tower()


    def encode_images(self, images):
        '''
        Encodes images into a feature space
        '''

        # Get features from vision tower
        selected_features = self.get_vision_tower()(images)

        # If we are using External Modular Fusion, then we need to project the features
        if self.config.layer_fusing_strategy == "E_M":
            # External Modular: Use a single projector for the final layer
            image_features_f = self.get_model().mm_projector_f(selected_features)      
            return image_features_f      
        if self.config.layer_fusing_strategy == "E_D":
            # Strategy selection for External Direct Fusion:
            # - For cases with fewer layers (e.g., 18, 3-18-23), we apply the Sparse Channel Integration (SCI) strategy,
            #   as proposed in the paper "Dense Connector for MLLMs" (Yao et al., 2024).
            # - For cases with more layers (e.g., latter, former, and all),
            #   we use the Dense Channel Integration (DCI) strategy, described in the same paper.
            # Reference:
            # Yao, H., Wu, W., Yang, T., Song, Y., Zhang, M., Feng, H., ... & Wang, J. (2024).
            # "Dense Connector for MLLMs." arXiv preprint arXiv:2405.13800.
            # Available at: https://arxiv.org/abs/2405.13800
            image_features = []
            image_features_2 = []
            if self.config.layer_using_strategy == '18':
                image_features_f = self.get_model().mm_projector_f(torch.cat([selected_features[0], selected_features[1]], dim=-1))
            if self.config.layer_using_strategy == '3-18' or self.config.layer_using_strategy == '3-18-23':
                image_features_f = self.get_model().mm_projector_f(torch.cat([selected_features[0], selected_features[1],selected_features[2]], dim=-1))
            if self.config.layer_using_strategy == 'former' or self.config.layer_using_strategy == 'latter':
                for i in range(0, 12):
                    image_features.append(selected_features[i])
                image_features = torch.stack(image_features, dim=0)
                image_features = torch.sum(image_features, dim=0) / 12
                image_features_f = self.get_model().mm_projector_f(torch.cat([image_features, selected_features[-1]], dim=-1))
            if self.config.layer_using_strategy == 'all':
                for i in range(0, 12):
                    image_features.append(selected_features[i])
                image_features = torch.stack(image_features, dim=0)
                image_features = torch.sum(image_features, dim=0) / 12
                for i in range(12,24):
                    image_features_2.append(selected_features[i])
                image_features_2 = torch.stack(image_features_2, dim=0)
                image_features_2 = torch.sum(image_features_2, dim=0) / 12
                image_features_f = self.get_model().mm_projector_f(torch.cat([image_features,image_features_2, selected_features[-1]], dim=-1))        
            if self.config.layer_using_strategy == 'last':
                image_features_f = self.get_model().mm_projector_f(selected_features[-1])
            return image_features_f
        
        else:
            # Internal Fusion: Project each layer separately
            image_features = []

            for idx, feature in enumerate(selected_features[:-1]): 
                image_features.append(self.get_model().mm_projectors[idx](feature))

            image_features_f = self.get_model().mm_projector_f(selected_features[-1])
            image_features.append(image_features_f)

            return image_features

    # Input Preparation
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        # Grab the vision tower
        vision_tower = self.get_vision_tower()
        # No vision tower or no images or no input ids -> returns early with minimal processing.
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if "E" in  self.config.layer_fusing_strategy:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
            else:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None
        # If images is a list or a 5D tensor (batch of images)
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            if "E" in  self.config.layer_fusing_strategy:
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
            else: # For Internal Fusion
                image_features_list = self.encode_images(concat_images) # Get the embeddings for each 
                split_sizes = [image.shape[0] for image in images]
                image_features = image_features_list[-1]

            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            # Flat: Flattens all patches into a single sequence
            # Spatial: Preserves spatial information between patches
            # Unpad: Removes padding to restore original aspect ratio
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if "E" in  self.config.layer_fusing_strategy:
                image_features = self.encode_images(images)
            else:
                image_features_list = self.encode_images(images)
                image_features = image_features_list[-1]
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


        # Combining Text and Image Features
        new_input_embeds = []
        new_labels = []
        image_token_mask = []

        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # Case 1: No images in the batch -> just use the text features
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                image_token_mask.append(torch.full((cur_input_embeds.shape[0],), False, device=cur_input_embeds.device, dtype=int))
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            # Case 2: Images in the batch -> process the text and image features separately
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # This creates a list like [-1, 5, 12, 20] where:
            # -1: Represents the position before the first token (the start of the sequence)
            # 5: The position of the first image token
            # 12: The position of the second image token
            # 20: The position after the last token (the end of the sequence)
            # [START] What is in [IMAGE] this image? [IMAGE] Please describe it. [END] 

            
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            # Embedding Creation: 1)  Creates embeddings for the text tokens. 2 Splits the embeddings to match the original segments
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # Combining Text and Image Embeddings
            # 1) Alternates between text embeddings and image features
            # 2) Creates labels for each segment
            # 3) Creates a mask to identify image tokens
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_image_token_mask = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_image_token_mask.append(torch.full((cur_input_embeds_no_im[i].shape[0],), False, device=cur_labels.device, dtype=cur_labels.dtype))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_image_token_mask.append(torch.full((cur_image_features.shape[0],), True, device=cur_labels.device, dtype=cur_labels.dtype))
                # Embeddings for [START] What is in
                # Features for the first image
                # Embeddings for this image?
                # Features for the second image
                # Embeddings for Please describe it. [END]
                # Walk through the example:
                    # 0: [START]
                    # 1: What
                    # 2: is
                    # 3: in
                    # 4: [IMAGE]  <-- First image token at position 4
                    # 5: this
                    # 6: image?
                    # 7: [IMAGE]  <-- Second image token at position 6
                    # 8: Please
                    # 9: describe
                    # 10: it.
                    # 11: [END]
                    # Resulting in: [-1, 4, 6, 11]

# **Input sequence:**
# ```
# [START] What is in [IMAGE] this image? [IMAGE] Please describe it. [END]
# ```

# **With two images:**
# - Image 1: A cat
# - Image 2: A dog

# **Processing steps:**

# 1. **Find image token positions:**
#    - `[-1, 5, 12, 20]` (assuming positions 5 and 12 are image tokens)

# 2. **Split text segments:**
#    - Segment 1: `[START] What is in`
#    - Segment 2: `this image?`
#    - Segment 3: `Please describe it. [END]`

# 3. **Create text embeddings:**
#    - Embeddings for Segment 1
#    - Embeddings for Segment 2
#    - Embeddings for Segment 3

# 4. **Alternate text and images:**
#    - Embeddings for Segment 1
#    - Features for Image 1 (cat)
#    - Embeddings for Segment 2
#    - Features for Image 2 (dog)
#    - Embeddings for Segment 3

# 5. **Create labels:**
#    - Labels for Segment 1
#    - `IGNORE_INDEX` for Image 1
#    - Labels for Segment 2
#    - `IGNORE_INDEX` for Image 2
#    - Labels for Segment 3

# 6. **Create image token mask:**
#    - `False` for Segment 1
#    - `True` for Image 1
#    - `False` for Segment 2
#    - `True` for Image 2
#    - `False` for Segment 3

            # Final Embedding Creation: 1) Moves embeddings to the correct device 2) Concatenates all embeddings, labels, and masks
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_image_token_mask = torch.cat(cur_image_token_mask)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            image_token_mask.append(cur_image_token_mask)
        # Truncates long sequences and pads short ones
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            image_token_mask = [x[:tokenizer_model_max_length] for x in image_token_mask]

        # Padding Prep
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        new_image_token_mask_padded = []

        # Padding Application
        for i, (cur_new_embed, cur_new_labels,cur_image_token_mask) in enumerate(zip(new_input_embeds, new_labels,image_token_mask)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                # Left Padding
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

                    new_image_token_mask_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_image_token_mask
                ), dim=0))
                    
            else:
                # Right Padding
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))

                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

                    new_image_token_mask_padded.append(torch.cat((
                    cur_image_token_mask, torch.zeros((max_len - cur_len), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                    ), dim=0))

        # Final Stacking and Return
        # 1. Stacks padded tensors: Combines all padded tensors into batches
        # 2. Restores original values: Handles cases where inputs were None
        # 3. Returns different values based on fusion strategy:
        #   - External Fusion (E_M, E_D): Returns a simpler set of values
        #   - Internal Fusion (I_M, I_D): Returns additional values for layer-specific features  
        
        # 1. Stack
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_image_token_mask = torch.stack(new_image_token_mask_padded,dim=0)
        
        # 2. Restore
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        if _position_ids is None:
            position_ids = None

        # 3. Return
        if "E" in  self.config.layer_fusing_strategy:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features,new_image_token_mask

        else:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_features_list[:-1], image_features_list[-1],new_image_token_mask


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
