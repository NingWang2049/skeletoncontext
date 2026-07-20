import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from model.modeling_lxmert import LxmertConfig, LxmertXLayer
from model.modeling_bert import BertConfig, BertModel, BertOnlyMLMHead

from model.PartDouple import JointImportanceDetermination
from model.PromptModel import PromptModel

import json
import random
import numpy as np
import math

import sys
sys.path.append("./model/Temporal_shift/")

from cuda.shift import Shift


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(25*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        index_array = np.empty(25*in_channels).astype(np.int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(25*out_channels).astype(np.int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        cube_feature = x.view(N, M, c_new, -1, V)
        cube_feature = cube_feature.mean(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x), x, cube_feature

class ModelMatch(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(ModelMatch, self).__init__()
        # pretraining model
        self.feature_extractor = Model(num_class, num_point, num_person, graph, graph_args, in_channels)
        for p in self.parameters():
            p.requires_grad = False
        
        if num_class == 60 or num_class == 120:
            # load context and joint text
            with open('semantics/ntu/context_from_descriptions.json', 'r', encoding='utf-8') as f:
                context_text = json.load(f)
                self.context_text = []
                for _, value in context_text.items():
                    self.context_text.append(value)
            with open('semantics/ntu/joint_from_descriptions.json', 'r') as f:
                self.joint_text = json.load(f)
        else:
            # load context and joint text
            with open('semantics/pku/context_from_descriptions.json', 'r', encoding='utf-8') as f:
                context_text = json.load(f)
                self.context_text = []
                for _, value in context_text.items():
                    self.context_text.append(value)
            with open('semantics/pku/joint_from_descriptions.json', 'r') as f:
                self.joint_text = json.load(f)
        
        # joint branch
        parts_map = {
            "head":[2, 3],
            "hand":[7, 11, 21, 22, 23, 24],
            "arm":[4, 5, 6, 8, 9, 10],
            "hip":[0, 1, 12, 16],
            "leg":[12, 13, 14, 16, 17, 18],
            "foot":[14, 15, 18, 19]
        }
        self.kg_joint = []
        for _,parts in self.joint_text.items():
            part_joints = [0] * 25
            for part in parts:
                for i in parts_map[part]:
                    part_joints[i] = 1
            self.kg_joint.append(torch.tensor(part_joints))
        self.kg_joint = torch.stack(self.kg_joint)
        self.jid = JointImportanceDetermination(256, 256, num_point)
        self.spatial_project = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU()
        )

        # prompt branch
        self.in_channels = 256        
        self.norm_q, self.norm_k, self.norm_v = nn.LayerNorm(self.in_channels), nn.LayerNorm(self.in_channels), nn.LayerNorm(self.in_channels)
        self.to_q = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.to_k = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.to_v = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.act=nn.ReLU()
        self.temporal_project = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU()
        )
        self.prompt_text = "Environment: [MASK] | Manipulable object: [MASK] | Target object: [MASK]"
        self.prompt_encoder = PromptModel()
    
    def get_context_data(self, target, epoch=0, max_epoch=1):
        prompt_text = self.prompt_text
        texts = []
        mask_texts = []
        mask_words = []
        L = prompt_text.count("[MASK]") - epoch*prompt_text.count("[MASK]")//max_epoch - 1
        prompt_parts = prompt_text.split(" | ")
        all_indices = list(range(prompt_text.count("[MASK]")))
        selected_indices = random.sample(all_indices, L) 
        
        for t in target:
            t = random.randint(0, len(self.context_text)-1)
            context_text = self.context_text[t][random.randint(0, 9)]
            context_text = context_text[:1] + context_text[2:3] + context_text[4:]
            random.shuffle(context_text)
            format_str = prompt_text.replace("[MASK]", "{}")
            texts.append(format_str.format(*context_text))
            
            mask_parts = []
            for i in range(prompt_text.count("[MASK]")):
                if i in selected_indices:
                    mask_part = prompt_parts[i].replace("[MASK]", context_text[i])
                else:
                    mask_part = prompt_parts[i]
                mask_parts.append(mask_part)
            mask_text = " | ".join(mask_parts)
            mask_texts.append(mask_text)
            
            mask_word = [context_text[i] for i in all_indices if i not in selected_indices]
            mask_words.append(mask_word)
        
        return texts, mask_texts, mask_words

    def forward(self, x, spatial_round, temporal_round, bert_lhs=None, target=None, epoch=0, max_epoch=0, vis=False):
        _, _, cube_feature = self.feature_extractor(x)  # n, 256, 16, 25
        cube_feature = cube_feature.permute(0, 2, 3, 1) # B, T, V, C
        # joint brach
        if target is None:
            weights, _ = self.jid(cube_feature)
        else:
            k_gt = self.kg_joint[target].to(cube_feature.device)  # B, V
            weights, loss_calib = self.jid(cube_feature, k_gt)
        sp = cube_feature * weights
        sp_proj_list = F.normalize(self.spatial_project(sp.mean(2).mean(1)), p=2, dim=1)
        
        # prompt branch
        feature = cube_feature.mean(1) # b,v,c
        q, k, v = self.norm_q(feature), self.norm_k(feature), self.norm_v(feature)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        diff = q.unsqueeze(1) - k.unsqueeze(2) # b,v,v,c
        diff = self.act(diff)
        diff = F.softmax(diff, dim=1)
        diff_feature = torch.einsum('bvd,bvwd->bwd', v, diff).contiguous()
        temporal_feature = self.temporal_project((diff_feature + feature).mean(1))
        # temporal_feature = self.temporal_project(cube_feature.mean(2).mean(1))
        if target is None:
            prompt_texts = [self.prompt_text for i in range(temporal_feature.shape[0])]
            # bert_lhs = bert_lhs.to(temporal_feature.device).expand(temporal_feature.shape[0], -1, -1)
            loss_mask, tp, slot_text_embs = self.prompt_encoder(temporal_feature, texts=prompt_texts)
            tp_proj_list = F.normalize(tp, p=2, dim=1)
            # semantic 
            spatial_fg_norm = F.normalize(spatial_round, p=2, dim=-1)   # 55(5) 10 768
            temporal_fg_norm = F.normalize(temporal_round, p=2, dim=-1)   # 55(5) 10 768
            # multiply
            logits_spatial_fg = torch.einsum('nd,ckd->nck', sp_proj_list, spatial_fg_norm).topk(10, dim=2)[0].mean(2)
            logits_temporal_fg = torch.einsum('nd,ckd->nck', tp_proj_list, temporal_fg_norm).topk(10, dim=2)[0].mean(2)
            if vis:
                return logits_spatial_fg*0.1, logits_temporal_fg*0.1, slot_text_embs
            else:
                return logits_spatial_fg*0.1, logits_temporal_fg*0.1
        else:
            texts, mask_texts, mask_words = self.get_context_data(target, epoch, max_epoch)
            loss_mask, tp, slot_text_embs = self.prompt_encoder(temporal_feature, texts, mask_texts, mask_words)
            tp_proj_list = F.normalize(tp, p=2, dim=1)
            
            # semantic 
            spatial_fg_norm = F.normalize(spatial_round, p=2, dim=-1)   # 55(5) 10 768
            temporal_fg_norm = F.normalize(temporal_round, p=2, dim=-1)   # 55(5) 10 768
            # multiply
            logits_spatial_fg = torch.einsum('nd,ckd->nck', sp_proj_list, spatial_fg_norm).topk(10, dim=2)[0].mean(2)
            logits_temporal_fg = torch.einsum('nd,ckd->nck', tp_proj_list, temporal_fg_norm).topk(10, dim=2)[0].mean(2)
            
            return logits_spatial_fg*0.1, logits_temporal_fg*0.1, loss_mask, loss_calib
