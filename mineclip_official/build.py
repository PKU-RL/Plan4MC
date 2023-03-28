import torch
from torch import nn
import numpy as np

from .modules import VisionTransformer,GPT,TemporalTransformer,AdapterHead

class videoCLIP(nn.Module):
    def __init__(self,
                 image_encoder: VisionTransformer, #| VisionTransformer_after_freeze,
                 text_encoder: GPT ,#| GPT_after_freeze,
                 temporal_encoder:TemporalTransformer,
                 reward_adapter:AdapterHead,
                 logit_scale:nn.Parameter
                 ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temporal_encoder = temporal_encoder
        self.reward_adapter = reward_adapter
        self.logit_scale = logit_scale
    
    def forward(self,videos,texts):
        videos = torch.as_tensor(videos,dtype=torch.float)
        images_embeddings = self.image_encoder(videos)
        text_embeddings = self.text_encoder(texts)
        video_embeddings = self.temporal_encoder(images_embeddings)
        adapted_video,adapted_text = self.reward_adapter(video_embeddings,text_embeddings)
        
        video_features = adapted_video / adapted_video.norm(dim=1, keepdim=True)
        text_features = adapted_text / adapted_text.norm(dim=1, keepdim=True)
        
        return self.logit_scale.exp()*video_features,text_features

    def get_layer(self,layer_idx):
        if layer_idx < 0:
            raise RuntimeError
        if layer_idx == 0:
            return self.reward_adapter,self.logit_scale
        elif layer_idx == 1:
            return (self.temporal_encoder,)
        elif layer_idx == 2:
            return self.image_encoder.ln_post,self.image_encoder.projection,self.text_encoder.ln_final,self.text_encoder.projection,
        elif (layer_idx-3) < self.image_encoder._layers:
            return self.image_encoder.blocks[-(layer_idx-2)],self.text_encoder.blocks[-(layer_idx-2)]
        else:
            return self.image_encoder.pos_embed,self.image_encoder.cls_token,self.image_encoder.conv1,self.image_encoder.ln_pre,\
                   self.text_encoder.pos_embed,self.text_encoder.token_embedding

    @torch.no_grad()
    def clamp_logit_scale(self, value=100):
        """
        Follow OpenAI CLIP paper's trick to prevent training instability (sec 2.5)
        """
        self.logit_scale.data.clamp_(-np.log(value), np.log(value))

def build_pretrain_model(image_config,text_config,temporal_config,adapter_config,state_dict=None)->videoCLIP:
    image_encoder = VisionTransformer(resolution=image_config['resolution'],patch_size=image_config['patch_size'],
                                      width=image_config['width'],layers=image_config['layers'],
                                      heads=image_config['heads'],output_dim=image_config['output_dim'])
    
    text_encoder = GPT(embed_dim=text_config['embed_dim'],context_length=text_config['context_length'],vocab_size=text_config['vocab_size'],
                       layers=text_config['layers'],width=text_config['width'],heads=text_config['heads'])
    
    temporal_encoder = TemporalTransformer(input_dim=temporal_config['input_dim'],depth=temporal_config['depth'],num_heads=temporal_config['num_heads'],
                                           max_seq_len=temporal_config['video_seq_len'],ff_glu=True,ff_swish=True)
    
    reward_adapter = AdapterHead(adapter_config['video_layers'],adapter_config['text_layers'],adapter_config['feature_dim'])

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    model = videoCLIP(image_encoder,text_encoder,temporal_encoder,reward_adapter,logit_scale)  
    
    if not state_dict is None :
        state_dict_back = model.state_dict()
        state_dict_back.update(state_dict)
        model.load_state_dict(state_dict_back)
    
    return model
