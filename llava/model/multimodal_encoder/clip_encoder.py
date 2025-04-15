import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.layer_using_strategy = args.layer_using_strategy
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True



    def feature_select(self, image_forward_outs):

        selected_features = []
        if self.layer_using_strategy == '18':
            select_layer = [18,23]
        if self.layer_using_strategy == '3-18':
            select_layer = [3,18,23]    
        if self.layer_using_strategy == '3-18-23':
            select_layer = [3,18,23,23]
        if self.layer_using_strategy == 'former':                 
            select_layer = [1,2,3,4,5,6,7,8,9,10,11,12,23]
        if self.layer_using_strategy == 'latter':
            select_layer = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,23]
        if self.layer_using_strategy == 'all':
            select_layer = [1,2,3,4,5,6,7,8,9,10,11,12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,23]
        if self.layer_using_strategy == 'last':
            # Only use the last layer output
            select_layer = [-1]

        for layer_index in select_layer:
            if layer_index == -1:
                # Use the last hidden state directly
                layer_features = image_forward_outs.last_hidden_state
            else:
                layer_features = image_forward_outs.hidden_states[layer_index]
            if self.select_feature == 'patch':
                layer_features = layer_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                layer_features = layer_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            selected_features.append(layer_features)


        return selected_features
    

    def forward(self, images):
        image_features = []

        if isinstance(images, list): 
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                selected_features = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append([feature.to(image.dtype) for feature in selected_features])
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            selected_features = self.feature_select(image_forward_outs)
            image_features = [feature.to(images.dtype) for feature in selected_features]

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
    
    @property
    def num_patches_vistoken(self):
        return (self.config.image_size // self.config.patch_size) * (self.config.image_size // self.config.patch_size) + 2

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
