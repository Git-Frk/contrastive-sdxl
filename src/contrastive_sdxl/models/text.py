import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

def load_text_encoders(pretrained_model_name_or_path="stabilityai/sdxl-turbo", device="cuda", revision=None):

    tokenizer_1 = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, 
                                                subfolder="tokenizer", 
                                                revision=revision)
    
    tokenizer_2 = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, 
                                                subfolder="tokenizer_2", 
                                                revision=revision)
    
    text_encoder_1 = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, 
                                                   subfolder="text_encoder").to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, 
                                                                 subfolder="text_encoder_2").to(device)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    return (tokenizer_1, tokenizer_2), (text_encoder_1, text_encoder_2)


@torch.no_grad()
def encode_prompt(caption, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2):
    outputs = []

    for tokenizer, text_encoder in ((tokenizer_1, text_encoder_1), (tokenizer_2, text_encoder_2)):
        tokens = tokenizer(caption, 
                           max_length=tokenizer.model_max_length, 
                           padding="max_length",
                           truncation=True, 
                           return_tensors="pt").input_ids
        tokens = tokens.to(next(text_encoder.parameters()).device)
        outputs.append(text_encoder(tokens, output_hidden_states=True))

    out_1, out_2 = outputs
    fixed_prompt_embeds = torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1)
    fixed_pooled_embeds = out_2[0]
    
    return fixed_prompt_embeds, fixed_pooled_embeds
