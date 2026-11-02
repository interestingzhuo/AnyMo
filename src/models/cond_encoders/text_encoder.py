import torch
# import clip
import torch.nn as nn
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from pathlib import Path
from transformers import T5Config, T5EncoderModel, T5Tokenizer

@MODELS.register_module()
class TextEncoder(BaseModel):
    def __init__(self, 
                 text_encoder_name = 'clip', 
                 max_text_length = 150,
                 output_dim = 512,
                 checkpoint_root = 'checkpoints',
                 pretrained = True,
                 local_files_only = False,
                 **kwargs):
        super().__init__()
        self.text_encoder_name = text_encoder_name
        self.max_text_length = max_text_length
        assert self.text_encoder_name in ['clip', 'flan-t5-xxl', 'flan-t5-xl']
        if self.text_encoder_name == 'clip':
            # raise NotImplementedError
            text_encoder = clip.load("checkpoints/ViT-B-32.pt", device='cuda', jit=False)[0]  
            clip.model.convert_weights(text_encoder) # type: ignore 
            self.text_encoder = text_encoder
            self.text_dim = 512
        elif self.text_encoder_name == 'flan-t5-xxl':
            path = str(Path(checkpoint_root) / self.text_encoder_name)
            self.tokenizer = T5Tokenizer.from_pretrained(path, local_files_only=local_files_only)
            self.text_encoder = (T5EncoderModel.from_pretrained(path, local_files_only=local_files_only)
                                 if pretrained else T5EncoderModel(T5Config.from_pretrained(path, local_files_only=local_files_only)))
            self.text_dim = 4096
        elif self.text_encoder_name == 'flan-t5-xl':
            path = str(Path(checkpoint_root) / self.text_encoder_name)
            self.tokenizer = T5Tokenizer.from_pretrained(path, local_files_only=local_files_only)
            self.text_encoder = (T5EncoderModel.from_pretrained(path, local_files_only=local_files_only)
                                 if pretrained else T5EncoderModel(T5Config.from_pretrained(path, local_files_only=local_files_only)))
            self.text_dim = 2048
        else:
            raise NotImplementedError
        
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        
        self.projector = nn.Sequential(
            nn.Linear(self.text_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, texts, **kwargs): # type: ignore
        '''
        texts: list[str], len B
        return: 
            condition_feature: tensor, B, n, c; 
            condition_mask: tensor, B, n
        '''
        with torch.no_grad():
            if self.text_encoder_name == 'clip':
                text_tokens = clip.tokenize(texts, truncate=True).cuda() # B, 77
                text_features = self.text_encoder.encode_text(text_tokens).unsqueeze(1)#.float() # B, 1, c
                text_mask = torch.ones((text_features.shape[0], text_features.shape[1])).cuda().int()
            elif self.text_encoder_name in ['flan-t5-xxl', 'flan-t5-xl']:
                text_tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                text_mask = text_tokens.attention_mask.cuda() # B, L
                text_features = self.text_encoder(
                    input_ids=text_tokens.input_ids.cuda(),
                    attention_mask=text_tokens.attention_mask.cuda(),
                    output_hidden_states=False
                ).last_hidden_state#.float() 
            else:
                raise NotImplementedError

        if text_features.shape[1] > self.max_text_length:
            text_features = text_features[:, :self.max_text_length, :]
            text_mask = text_mask[:, :self.max_text_length]

        text_features = self.projector(text_features)
        return text_features, text_mask
