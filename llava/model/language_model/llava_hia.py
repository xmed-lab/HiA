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


from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings, ModelOutput

from transformers import AutoConfig, AutoModelForCausalLM 
from .modeling_llama_hia import LlamaConfig, LlamaModel, LlamaForCausalLM


from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_hia"


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
    
        assert config.vocab_size == self.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.condition_linear = nn.Linear(2048, config.hidden_size)
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
        hr_images: Optional[torch.FloatTensor] = None,
        image_features_index: Optional[list] = None,
        return_dict: Optional[bool] = None,

    ) -> Union[Tuple, CausalLMOutputWithPast]:
       

        if inputs_embeds is None:
            
             (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_features_index,
                hr_image_features
            ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values, labels, images,
                    hr_images = hr_images,
                    image_features_index = image_features_index
                )
        hr_image_features = hr_image_features.transpose(2,3)
        hr_image_features = self.condition_linear(hr_image_features)
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            con_inputs_embeds= hr_image_features,
            image_features_index = image_features_index,
        )





    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        image_features_index = outputs.image_features_index

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs, model_kwargs = model_kwargs, is_encoder_decoder=is_encoder_decoder, standardize_cache_format=standardize_cache_format,
        )

        model_kwargs["image_features_index"] = image_features_index
       
        return model_kwargs
   
    

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        hr_images = kwargs.pop("hr_images", None)
        image_features_index = kwargs.pop("image_features_index", None)

      
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if hr_images is not None:
            _inputs['hr_images'] = hr_images
        if image_features_index is not None:
            _inputs['image_features_index'] = image_features_index
      
        return _inputs

AutoConfig.register("llava_hia", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
