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

from transformers import AutoConfig, AutoModelForCausalLM
                         
from .modeling_llama import LlamaModel, LlamaForCausalLM
from .configuration_llama import LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.mm_utils import unbiased_cka, cosine_similarity

import wandb
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

        # if compute_cka:
        # Then compare image_features 
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        original_output_hidden_states = output_hidden_states
        output_hidden_states = True if compute_cka else (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features_list = None
        image_features_f = None
        if "E" in self.config.layer_fusing_strategy:
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
            cka_similarities = {}
            cosine_similarities = {}

            for layer_idx, layer_hidden_state in enumerate(outputs.hidden_states):  # [batch, seq_len, hidden_dim]
                layer_ckas = []
                layer_cosines = []
                # So, we actually dont need to do this per sample. We dont even need to do this per batch.
                # We need to do this per layer. 

                # Loop over each sample in the batch
                hl_bsz = layer_hidden_state.shape[0]
                for sample_idx in range(hl_bsz):
                    # Get text embeddings for this batch
                    image_mask = image_token_mask[sample_idx].bool()
                    text_mask = ~image_mask                    
                    # Extract embeddings
                    text_embeds = layer_hidden_state[sample_idx][text_mask]    # [n_text_tokens, hidden_dim]
                    image_input_embeds = inputs_embeds[sample_idx][image_mask]  # [n_image_patches, hidden_dim]
                    # both hidden_state and input_embeds have the same shape. one is raw input, the other is processed
                    # But text embeds and image embeds have different shapes.
                    # So we need to align them before computing CKA? 
                    # Would it be better to align the image embeds to the text embeds?

                    # Only compute CKA if we have both text and image embeddings
                    if text_embeds.shape[0] > 0 and image_input_embeds.shape[0] > 0:
                        # try:        
                        # print(f"Text embeds shape: {text_embeds.shape}")
                        # print(f"Image embeds shape: {image_embeds.shape}")
                        # print(f"Text embeds dtype: {text_embeds.dtype}")
                        # print(f"Image embeds dtype: {image_embeds.dtype}")
                        cka_val = unbiased_cka(text_embeds, image_input_embeds)
                        layer_ckas.append(cka_val.item())
                        cos_sim = cosine_similarity(text_embeds, image_input_embeds)
                        layer_cosines.append(cos_sim)
                        # except Exception as e:
                        #     print(f"Warning: CKA computation failed for sample {sample_idx} in layer {layer_idx}: {e}")
                        #     layer_ckas.append(None)
                    else:
                        layer_ckas.append(None)
                        layer_cosines.append(None)
                # Aggregate layer CKA values per sample in batch via mean
                # So this is: Per each layer, we have a list of CKA values for each sample in the batch.
                # That means we compare each img_embed to text_embed for each sample in the batch --> Got CKA for that
                valid_layer_ckas = [sample_ckas for sample_ckas in layer_ckas if sample_ckas is not None] # [bsz] -> Just ckas that were able to be computed
                if valid_layer_ckas:
                    layer_mean = sum(valid_layer_ckas) / len(valid_layer_ckas)
                else:
                    layer_mean = None
                cka_similarities[f'layer_{layer_idx}'] = layer_mean

                valid_cosines = [val for val in layer_cosines if val is not None]
                cosine_mean = sum(valid_cosines) / len(valid_cosines) if valid_cosines else None
                cosine_similarities[f'layer_{layer_idx}'] = cosine_mean     
                
                if torch.distributed.get_rank() == 0:
                    if layer_mean is not None:
                        wandb.log({f"cka/layer_{layer_idx}": layer_mean})
                    if cosine_mean is not None:
                        wandb.log({f"cosine/layer_{layer_idx}": cosine_mean})
# # After collecting all layer CKAs, aggregate across ranks
# if torch.distributed.is_initialized():
#     # Gather CKA scores from all ranks
#     gathered_ckas = [None] * torch.distributed.get_world_size()
#     torch.distributed.all_gather_object(gathered_ckas, layer_ckas)
    
#     # Combine valid CKA scores (remove Nones)
#     valid_ckas = [score for rank_ckas in gathered_ckas 
#                  for score in rank_ckas if score is not None]
    
#     # Compute average CKA if we have valid scores
#     if valid_ckas:
#         avg_cka = sum(valid_ckas) / len(valid_ckas)
        
#         # Log only on rank 0
#         if torch.distributed.get_rank() == 0:
#             wandb.log({
#                 f"cka/layer_{layer_idx}": avg_cka,
#             })


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
