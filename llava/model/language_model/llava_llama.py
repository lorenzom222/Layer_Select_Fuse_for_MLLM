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

import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
                         
from .modeling_llama import LlamaModel, LlamaForCausalLM
from .configuration_llama import LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.mm_utils import compute_cka_unbiased, compute_mmd_linear, compute_cosine_similarity

import wandb
from llava.utils.token_utils import check_and_reconstruct_identical_tokens

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        images_features: Optional[List[torch.FloatTensor]] = None,
        image_token_mask: Optional[torch.FloatTensor] = None,
        compute_cka: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Check if we're in vanilla LLM mode - if so, skip all image processing
        # vanilla_llm_mode = getattr(self.config, 'vanilla_llm_mode', False)
        # if vanilla_llm_mode:
        #     # In vanilla LLM mode, just set all image stuff to None
        #     image_features_list = None
        #     image_features_f = None
        #     if "E" in self.config.layer_fusing_strategy:
        #         image_features_f = images_features[-1]
        #     else:
        #         image_features_list = images_features

                

            
        # If not in vanilla mode, continue with multimodal processing
        # if compute_cka:
        # Then compare image_features 
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        original_output_hidden_states = output_hidden_states
        output_hidden_states = True if compute_cka else (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features_list = None
        image_features_f = None
        if "E" in self.config.layer_fusing_strategy:
            # - image_features_f is always the last layer's features [batch_size, num_patches, hidden_dim]
            # - image_features_list contains all layers except the last (for Internal Fusion) List of [batch_size, num_patches, hidden_dim]
            # - images_features is the final variable that gets used in the model, which is either:
            #   - Just image_features_f for External Fusion
            #   - image_features_list + [image_features_f] for Internal Fusion
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    image_features_f,
                    image_token_mask,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )
            if images_features is None:
                images_features = image_features_f
        else:
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    image_features_list,
                    image_features_f,
                    image_token_mask,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )

            # NOTE:
            # THIS MIGHT BE A LIST OF TENSORS
            # For now, assume we are just using last layer from ViT
            # If we wanna do interal ViT selection, we handle this as a list
            if images_features is None and image_features_list is not None:
                images_features = image_features_list + [image_features_f]



        outputs = self.model(
            image_token_mask=image_token_mask,
            images_features=images_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # TODO: 
        # 1. CKA for each batch separately, since it's not meaningful to compute CKA across unaligned (image, prompt) pairs
        # 2. LLM input should be text only, no image ; when computing CKA. 
        #    - dummy image tokens would change (text) hidden states. It may change the distrubtions. So A) we either just training vanilla LLM, or B) we need to find best dummy tokens to not disturb the text hidden states.
        # 3. Right we are comparing not just image embeds to text embeds, but it's doing for across the batch. So effectively we are comparing say like 5 image embeds to 5 text embeds where 4/5 have nothing to do with each other. 
        # EX: A batch of 5 images, 4 are unrelated to the text prompt. More specific Ex: There is 1 image of a dog tied to 1 prompt about a dog + answer and this is one input in a batch, but there are 4 other inputs in the batch that have nothing to do with the dog. How do we fix this?
        # NOTE:
        # Currently it's comparing input image embeddings to hidden state image embeddings from each layer (which is not what we want).
        # We want to compare the image input embeddings (from the visual encoder) to 
        #   the !!text!! token hidden states, layer by layer.
        # We dont need to worry about patch or special tokens since you need to init them with flags and they are only used for image
        # So we can just mask them out.
    
        cka_similarities = None
        if compute_cka and outputs.hidden_states is not None and image_token_mask is not None:
            # print("\n=== Batch Size Debug Info ===")
            # print(f"Input batch size: {inputs_embeds.shape[0] if inputs_embeds is not None else 'None'}")
            # print(f"Hidden states batch size: {outputs.hidden_states[0].shape}")
            # print(f"Image token mask batch size: {image_token_mask.shape}")
            # print(f"Attention mask batch size: {attention_mask.shape}")
            # print(f"Image features batch size: {images_features.shape}")
            # print("==========================\n")
            mmd_vals = []
            cos_sim = []

            for layer_idx, layer_hidden_state in enumerate(outputs.hidden_states):
                # print(f"\n=== Layer {layer_idx} Debug Info ===")
                # print(f"Layer hidden state shape: {layer_hidden_state.shape}")
                # print(f"Image token mask shape: {image_token_mask.shape}")
                # print(f"Attention mask shape: {attention_mask.shape}")

                # ############################################################
                
                # text_token_mask = (image_token_mask == 0) & (attention_mask == 1)
                # print(f"Text token mask: {text_token_mask}")
                # print(f"Text token mask shape: {text_token_mask.shape}")
                # text_token_indices = torch.where(text_token_mask)
                
                # print(f"Text token indices shape: {text_token_indices}")

                # text_embeds = layer_hidden_state[text_token_indices]
                # image_embeds = images_features
                # print(f"text_embeds shape: {text_embeds.shape}")
                # print(f"image_embeds shape: {image_embeds.shape}")
                # ############################################################

                image_embeds_flat = []
                text_embeds_flat = []
                text_embeds_plain = []
                image_embeds_plain = images_features
                batch_size = layer_hidden_state.shape[0]
                

                
                for sample_idx in range(batch_size):
                    # print(f"\n-- Sample {sample_idx} --")
                    image_embeds = images_features[sample_idx] # Just pluck straight from the features
                    image_flat = image_embeds.reshape(-1)      # [N_img * D]
                    image_embeds_flat.append(image_flat)
                    cur_hidden_sample  = layer_hidden_state[sample_idx]   # [T, D]
                    cur_image_mask = image_token_mask[sample_idx]     # [T]
                    cur_attention_mask= attention_mask[sample_idx]       # [T]

                    # Masks
                    text_token_mask = (cur_image_mask == 0)# & (cur_attention_mask == 1)
                    text_token_indices = torch.where(text_token_mask)
                    text_embeds = cur_hidden_sample[text_token_indices]
                    text_embeds_plain.append(text_embeds)
                    text_flat = text_embeds.reshape(-1)
                    text_embeds_flat.append(text_flat)

                    # if torch.allclose(text_embeds[0], text_embeds[1:], rtol=1e-5, atol=1e-5):
                    #     print("Identical text tokens found in sample %d" % sample_idx)
                    #     continue
                    
                    # # Decode text tokens for debugging/analysis
                    # if text_token_indices[0].numel() > 0 and torch.allclose(text_embeds[0], text_embeds[1:], rtol=1e-5, atol=1e-5):  # Check if we have any text tokens
                    #     print("Identical text tokens found in sample %d" % sample_idx)
                    #     # Load tokenizer for decoding
                    #     tokenizer = AutoTokenizer.from_pretrained(
                    #         self.config._name_or_path,  # Use the model's path
                    #         use_fast=False,
                    #         padding_side="right"
                    #     )
                    #     # Get the text embeddings from input_embeds
                    #     text_embeds = inputs_embeds[sample_idx][text_token_indices]
                    #     # Find closest token embeddings in vocabulary
                    #     token_embeddings = self.get_model().embed_tokens.weight
                    #     distances = torch.cdist(text_embeds, token_embeddings)
                    #     closest_token_ids = torch.argmin(distances, dim=1)
                    #     decoded_text = tokenizer.decode(closest_token_ids)
                    #     print(f"\nSample {sample_idx} text tokens: {closest_token_ids.tolist()}")
                    #     print(f"Sample {sample_idx} decoded text: {decoded_text}")
                
                    # print(f"Text embeds shape: {text_embeds.shape}")
                    # print(f"Image embeds shape: {image_embeds.shape}")
                    # print(f"Text flat shape: {text_flat.shape}")
                    # print(f"Image flat shape: {image_flat.shape}")
                    # print("==========================\n")
                    mmd_vals.append(compute_mmd_linear(text_embeds, image_embeds))
                    cos_sim.append(compute_cosine_similarity(text_embeds, image_embeds))

                image_embeds_flat = torch.stack(image_embeds_flat, dim=0)
                text_embeds_flat = torch.stack(text_embeds_flat, dim=0)
                text_embeds_plain = torch.stack(text_embeds_plain, dim=0)
                # print(f"Image embeds flat shape: {image_embeds_flat.shape}")
                # print(f"Text embeds flat shape: {text_embeds_flat.shape}")
                # print(f"Image embeds plain shape: {image_embeds_plain.shape}")
                # print(f"Text embeds plain shape: {text_embeds_plain.shape}")

                # print("==========================\n")
                # if torch.allclose(text_embeds_flat[0], text_embeds_flat[1:], rtol=1e-5, atol=1e-5):
                #     print("Identical text tokens found in text_embeds_flat")
                # cka_similarities = compute_cka_unbiased(
                #     text_embeds_flat, 
                #     image_embeds_flat,
                #     unbiased=True,
                #     model=self.get_model(),
                # )
                # print(f"CKA similarities: {cka_similarities}")
                # print("==========================\n")

                mmd_similarities = torch.tensor(mmd_vals).mean().item()
                cos_sim_similarities = torch.tensor(cos_sim).mean().item()
                # print(f"MMD similarities: {mmd_similarities}")
                # print("==========================\n")
                if torch.distributed.get_rank() == 0:
                    # if cka_similarities is not None:
                    #     wandb.log({f"cka/layer_{layer_idx}": cka_similarities})
                    if mmd_similarities is not None:
                        wandb.log({f"mmd/layer_{layer_idx}": mmd_similarities})
                    if cos_sim_similarities is not None:
                        wandb.log({f"cos_sim/layer_{layer_idx}": cos_sim_similarities})
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            output = output + (cka_similarities,)
            if not original_output_hidden_states and hasattr(outputs, 'hidden_states'):
                 print("Warning: Requesting CKA without return_dict=True might lead to unexpected output tuple structure.")
                 pass
            return (loss,) + output if loss is not None else output

        output_obj = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if (original_output_hidden_states or compute_cka) else None,
            attentions=outputs.attentions,
        )
        if cka_similarities is not None:
            output_obj.cka_similarities = cka_similarities
            
        return output_obj


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        image_features_list = None

        if "E" in self.config.layer_fusing_strategy:
            if images is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                    image_features_f,
                    image_token_mask
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)

            if image_features_f != None:
                images_features = image_features_f
        else:
            if images is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                    image_features_list,
                    image_features_f,
                    image_token_mask
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)
            if image_features_list != None:
                images_features = image_features_list + [image_features_f]



        return super().generate(
            image_token_mask = image_token_mask,
            images_features = images_features,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_token_mask = kwargs.pop("image_token_mask", None)
        images_features = kwargs.pop("images_features", None)


        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        if image_token_mask is not None:
            inputs['image_token_mask'] = image_token_mask
        if images_features is not None:
            inputs['images_features'] = images_features


        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
