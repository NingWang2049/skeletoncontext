import torch
import torch.nn as nn

from transformers import BertTokenizer
from model.modeling_lxmert import LxmertConfig, LxmertXLayer
from model.modeling_bert import BertConfig, BertModel, BertOnlyMLMHead

class PromptModel(nn.Module):
    def __init__(self, max_len=40, num_layers=1):
        super(PromptModel, self).__init__()

        self.max_len = max_len
        self.num_layers = num_layers        
        self.criterion = nn.CrossEntropyLoss()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # freeze bert parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.mask_id = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

        self.config = BertConfig()
        self.cls = BertOnlyMLMHead(self.config)

        self.lxmert_config = LxmertConfig()
        self.lxmert_xlayer = LxmertXLayer(self.lxmert_config)

    def forward(self, visual_feats, texts=[], mask_texts = [], mask_words = [], bert_lhs=None):
        visual_feats = visual_feats.unsqueeze(1)  # [24, 1, 768] 
        if len(mask_texts) != 0:
            inputs = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                max_length = self.max_len,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids = True,
                return_attention_mask = True,
                add_special_tokens=True
            )
            label = inputs.input_ids.to(visual_feats.device)
            
            for idx, word in enumerate(mask_words):
                parts = mask_texts[idx].split('[MASK]')
                num_masks = len(parts) - 1
                replaced_parts = [parts[0]]
                for i in range(num_masks):
                    token_count = len(self.tokenizer.tokenize(word[i]))
                    replaced_mask = '[MASK]' * token_count
                    replaced_parts.append(replaced_mask)
                    replaced_parts.append(parts[i+1])
                mask_texts[idx] = ''.join(replaced_parts)

            inputs = self.tokenizer.batch_encode_plus(
                mask_texts,
                padding=True,
                max_length = self.max_len,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids = True,
                return_attention_mask = True,
                add_special_tokens=True
            )

            inputs.attention_mask = inputs.attention_mask.to(visual_feats.device)
            inputs.input_ids = inputs.input_ids.to(visual_feats.device)
            inputs.token_type_ids = inputs.token_type_ids.to(visual_feats.device)
            
            text_embedding = self.bert(
                input_ids=inputs.input_ids,
                token_type_ids=inputs.token_type_ids,
                attention_mask = inputs.attention_mask,
            )

            text_hidden_state = text_embedding[0] # [24, 100, 768]
            # lxmert的xlayer ，self attention and cross attention
            lang_feats = text_hidden_state   # [24,95,768]
            for i in range(self.num_layers):
                x_outputs = self.lxmert_xlayer(
                    lang_feats = lang_feats,
                    lang_attention_mask = None,  
                    visual_feats = visual_feats,
                    visual_attention_mask = None,
                    input_id = inputs.input_ids,
                    output_attentions=False,
                )
                lang_feats, visual_feats = x_outputs[:2]
            
            # compute mask_loss
            loss_mask = 0.
            output_mask = self.cls(lang_feats)
            label = self._build_mlm_labels(inputs.input_ids, label)
            for i in range(len(texts)):
                loss_mask = loss_mask + self.criterion(output_mask[i], label[i])
            loss_mask = loss_mask / len(texts)
            
            return loss_mask, visual_feats.squeeze(1),output_mask
        else:
            inputs = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                max_length = self.max_len,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids = True,
                return_attention_mask = True,
                add_special_tokens=True
            )
            inputs.attention_mask = inputs.attention_mask.to(visual_feats.device)
            inputs.input_ids = inputs.input_ids.to(visual_feats.device)
            inputs.token_type_ids = inputs.token_type_ids.to(visual_feats.device)
            
            if bert_lhs is not None:
                text_hidden_state = bert_lhs  # [24, 18, 768]
            else:
                text_embedding = self.bert(
                    input_ids=inputs.input_ids,
                    token_type_ids=inputs.token_type_ids,
                    attention_mask = inputs.attention_mask,
                )

                text_hidden_state = text_embedding[0] # [24, 100, 768]
            
            # lxmert的xlayer ，self attention and cross attention
            lang_feats = text_hidden_state   # [24,95,768]
            for i in range(self.num_layers):
                x_outputs = self.lxmert_xlayer(
                    lang_feats = lang_feats,
                    lang_attention_mask = None,  
                    visual_feats = visual_feats,
                    visual_attention_mask = None,
                    input_id = inputs.input_ids,
                    output_attentions=False,
                )
                lang_feats, visual_feats = x_outputs[:2]
            
            # output_mask = self.cls(lang_feats)
            
            return None, visual_feats.squeeze(1), lang_feats
    
    def _build_mlm_labels(self, masked_input_ids, full_input_ids):
        labels = full_input_ids.clone()
        labels[masked_input_ids != self.mask_id] = -100
        return labels
