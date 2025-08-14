import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from utils import objectives
import torch
import os
from sklearn.decomposition import PCA

import cn_clip.clip as clip
from cn_clip.clip import load_from_name
_clip_model = None
_clip_preprocess = None
from module.DataEncoder import VitalSigDataset
vsEncoder = VitalSigDataset()

def load_clip_once(device, cache_dir):
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        if os.path.exists(cache_dir) and not os.path.isdir(cache_dir):
            os.remove(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        _clip_model, _clip_preprocess = load_from_name(
            "ViT-B-16", device=device, download_root=cache_dir
        )
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_preprocess


class FeatureExtractor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1,
                 bidirectional: bool = False,#True
                 cell_type: str = 'lstm' 
                 ):
        super().__init__()
        assert cell_type in ('lstm', 'gru'), "cell_type must be 'lstm' or 'gru'"
        self.cell_type = cell_type
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_dim // self.num_directions
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:  # 'gru'
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        self.fc = nn.Linear(self.hidden_size * self.num_directions, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        rnn_out = self.rnn(x)
        if self.cell_type == 'lstm':
            out_seq, (h_n, c_n) = rnn_out
        else:
            out_seq, h_n = rnn_out
        last_out = out_seq[:, -1, :]  # [B, hidden_size * num_directions]
        return self.fc(last_out)

class VitalProj(nn.Module):
    def __init__(self, in_dim, out_dim, n_bins=10, emb_dim=None, hidden_dim=64):
        super().__init__()
        self.in_dim = in_dim
        # 如果没给 emb_dim，就先用一个 heuristic
        emb_dim = emb_dim or max(out_dim // in_dim, 1)
        # 每个特征一个 embedding table
        self.embs = nn.ModuleList([nn.Embedding(n_bins, emb_dim) for _ in range(in_dim)])
        # 把所有 feature embedding 拼起来后再做一次 MLP 映射到 out_dim
        self.post = nn.Sequential(
            nn.Linear(in_dim * emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.n_bins = n_bins

    def forward(self, X):
        # X: [B, in_dim], 连续值
        # 1) 离散化成 bin_id
        #    这里举例等宽分箱，实际可换成你自己的分箱逻辑
        with torch.no_grad():
            # 比如先把值映射到 [0, n_bins)
            # （你可改成按分位点或自定义边界）
            bin_ids = torch.clamp((X / X.abs().max(dim=0, keepdim=True).values) * (self.n_bins/2) 
                                  + (self.n_bins/2), 0, self.n_bins-1).long()
        # 2) 每个特征 lookup
        emb_list = []
        for i in range(self.in_dim):
            emb_list.append(self.embs[i](bin_ids[:, i]))  # [B, emb_dim]
        # 3) 拼接 & 后续 MLP
        emb_cat = torch.cat(emb_list, dim=1)              # [B, in_dim*emb_dim]
        return self.post(emb_cat)                        # [B, out_dim]


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
    def forward(self, x, y=None):
        if y is None:
            y = x
        attn_output, _ = self.attention(x, y, y)
        # Add & Norm
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        # Add & Norm
        x = residual + x
        x = self.norm(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim, dropout)
        
    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_att = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim, dropout)
        
    def forward(self, x, encoder_output):
        x = self.self_attention(x)
        x = self.cross_att(x, encoder_output)
        x = self.ffn(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

class Transformer(nn.Module):

    def __init__(self, args, input_dim, embed_dim, num_heads, 
                 hidden_dim, num_encoder_layers, num_decoder_layers, 
                 output_dim_s, output_dim_d, per_cls_vs, per_cls_cc,input_dim_vs,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.args = args
        self.enc = args.text_encoder.strip().lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.vs_group_sizes = list(vsEncoder.num_classes.values())
        self.output_dim_s = output_dim_s
        self.output_dim_d = output_dim_d
        self.per_cls_vs = per_cls_vs
        self.per_cls_cc = per_cls_cc

        self.VitalEmbed = FeatureExtractor(input_dim=input_dim_vs, 
                                           hidden_dim=args.hidden_dim, 
                                           output_dim=embed_dim, 
                                           num_layers=args.lstm_layers,
                                           bidirectional=True, cell_type='lstm')#lstm, gru
        
        self.encoder = TransformerEncoder(num_encoder_layers, embed_dim, 
                                          num_heads, hidden_dim, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, embed_dim, 
                                          num_heads, hidden_dim, dropout)
        self.cross_att1 = MultiHeadAttention(embed_dim, num_heads, dropout)#cc
        self.cross_att2 = MultiHeadAttention(embed_dim, num_heads, dropout)#vs

        self.resnet_vs = ResNet(input_dim=input_dim+input_dim_vs, output_dim=embed_dim)
        self.resnet_cc = ResNet1D(input_dim=input_dim+input_dim_vs, output_dim=embed_dim, base_model='resnet18',  pretrained=True)

        self.mlp = MLP(input_dim=input_dim, output_dim=embed_dim)
        self.resnet = ResNet(input_dim=input_dim, output_dim=embed_dim)
        self.textcnn = TextCNN(input_dim=input_dim, output_dim=embed_dim)
        # self.textresnet = TextResNet(input_dim=input_dim, output_dim=embed_dim)
        self.textresnet = TextResNet(emb_dim=input_dim, num_classes=embed_dim)
        self.linear_layer_cc = nn.Linear(input_dim, embed_dim)
        self.linear_layer_vs = nn.Linear(input_dim_vs, embed_dim)
        self.CEL = nn.CrossEntropyLoss()        
        self.task_set()  

        if self.args.CMF or self.args.vsEmbed:
            vs_embed_dim = embed_dim
            cc_embed_dim = embed_dim
        elif self.args.FusionEarly and not self.args.vsEmbed:
            vs_embed_dim = input_dim_vs
            cc_embed_dim = input_dim 
        elif not (self.args.FusionEarly and self.args.vsEmbed and self.args.backbone):
            vs_embed_dim = input_dim_vs
            cc_embed_dim = input_dim 
        else:
            vs_embed_dim = input_dim_vs
            cc_embed_dim = embed_dim


        self.fc_severity   = nn.Linear(vs_embed_dim, output_dim_s)
        self.fc_department = nn.Linear(cc_embed_dim, output_dim_d)

    def task_set(self): 
        loss_names = self.args.loss
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} loss(es)')

    def FusionModule(self, VS, ccs):
        device, dtype = VS.device, VS.dtype
        if self.args.backbone == 'TextCNN':
            cc_cls = ccs[:, 0, :]
        else:
            cc_cls = ccs
        CC = cc_cls.to(device)                     # (B, D2)
        eps = 1e-6
        VS_norm = (VS - VS.mean(dim=0)) / (VS.std(dim=0) + eps)
        CC_norm = (CC - CC.mean(dim=0)) / (CC.std(dim=0) + eps)
        
        VS_pca = PCA(n_components=self.args.n_comp) \
            .fit_transform(VS_norm.detach().cpu().numpy())
        CC_pca = PCA(n_components=self.args.n_comp) \
            .fit_transform(CC_norm.detach().cpu().numpy())

        VS_pca = torch.from_numpy(VS_pca).to(device=device, dtype=dtype)
        CC_pca = torch.from_numpy(CC_pca).to(device=device, dtype=dtype)
        Combined_vs = torch.cat([VS_norm, CC_pca], dim=1)  # (B, D1 + n_comp)
        Combined_cc = torch.cat([CC_norm, VS_pca], dim=1)  # (B, D2 + n_comp)
        def torch_standardize(x):
            mu = x.mean(dim=0, keepdim=True)
            sigma = x.std(dim=0, keepdim=True) + eps
            return (x - mu) / sigma

        VitalSign  = torch_standardize(Combined_vs)
        ChiefComp  = torch_standardize(Combined_cc)

        if self.args.backbone == 'TextCNN':
            ccs_tokens = ccs[:, 1:, :]
            VS_expand = VS_pca.unsqueeze(1).expand(-1, ccs_tokens.size(1), -1)
            CC_withVS = torch.cat([ccs_tokens, VS_expand], dim=2)
            ChiefComp = torch.cat([ChiefComp.unsqueeze(1), CC_withVS], dim=1)

        return VitalSign, ChiefComp


    def _apply_vs_group_mask(self, VS: torch.Tensor) -> torch.Tensor:
        """
        VS: (B, D_total) from one‑hot structure
        随机在 G=8 个指标里选 k=⌊rate·G⌋ 个，把对应的子区间全置零。
        """
        if not self.training or not getattr(self.args, 'use_vs_mask', False):
            return VS

        rate = self.args.vs_mask_rate
        G = len(self.vs_group_sizes)
        k = max(1, int(G * rate))

        # 随机挑 k 个指标下标
        idxs = torch.randperm(G)[:k].tolist()

        mask = torch.ones_like(VS)
        start = 0
        for g, sz in enumerate(self.vs_group_sizes):
            if g in idxs:
                mask[:, start:start+sz] = 0.0
            start += sz
        return VS * mask

    def extract_features(self, batch, cc_cls):
        vs0 = batch['VS']   
        if self.args.use_vs_mask:
            vs_masked = self._apply_vs_group_mask(vs0)
        else:
            vs_masked = vs0
        # if self.args.SFD:
        #     vs_masked = vsEncoder.SFD_encoder(vs_masked) 
        VitalSign, ChiefComp = self.FusionModule(vs_masked, cc_cls)
        return VitalSign.to(torch.float32), ChiefComp.to(torch.float32)

    def forward(self, batch):
        ret = dict()
        label1, label2 = batch['Level'],  batch['Dept_digit']

        if self.enc == 'clip' or self.enc == 'cn_clip' :
            text_tokens = batch['CC_tokens'].long()

            self.model, _ = load_clip_once(self.device, self.args.cache_clip)
            text_feats = self.model.encode_text(text_tokens)
            text_feats = F.normalize(text_feats, dim=-1)
            cc_cls = text_feats.float()#(bs, 512)
        else:
            cc_cls = batch['CC_tokens']

        if self.args.FusionEarly:# and not self.args.backbone == 'TextCNN':
            VitalSign, ChiefComp = self.extract_features(batch, cc_cls)
            VitalSign = VitalSign.to(self.device)
        else:
            VitalSign = batch['VS']
            # ChiefComp = F.normalize(cc_cls, dim=-1)
            ChiefComp = cc_cls
        # ************************************************************************************
        #                            backbone  
        # ************************************************************************************
        if self.args.backbone =='TextResNet':
            logit_cc = self.textresnet(ChiefComp)

        elif self.args.backbone == 'Transformer':
            cc_feat = self.linear_layer_cc(ChiefComp)
            enc_cc_cls = self.encoder(cc_feat)
            logit_cc = self.decoder(cc_feat, enc_cc_cls)

        elif self.args.backbone == 'TextCNN':
            logit_cc = self.textcnn(ChiefComp) 

        elif self.args.backbone == 'MLP':
            logit_cc = self.mlp(ChiefComp) 
        elif self.args.backbone == 'ResNet':
            logit_cc = self.resnet(ChiefComp.unsqueeze(1))   
                      
        else:
            logit_cc = self.linear_layer_cc(ChiefComp) if self.args.CMF else ChiefComp
            # logit_cc = self.resnet_cc(ChiefComp)  

        # VitalSign = F.normalize(VitalSign, dim=-1)  

        # logit_vs = self.VitalEmbed(VitalSign) if self.args.vsEmbed else VitalSign #LSTM

        if self.args.vsEmbed:
            logit_vs = self.VitalEmbed(VitalSign)
        elif self.args.CMF and not self.args.vsEmbed:
            logit_vs = self.linear_layer_vs(VitalSign)
        else:
            logit_vs = VitalSign


        if self.args.CMF: #logit_vs/cc:(bs,128);fusion_vs/cc:(bs,256);concat_vs/cc:(bs,256)
            if self.args.backbone == 'TextCNN':
                ChiefComp = ChiefComp[:, 0, :]
            fusion_feat_raw = torch.cat([VitalSign, ChiefComp], dim=1)
            logit_cv1 = self.resnet_vs(fusion_feat_raw.unsqueeze(1))
            # logit_cv2 = self.resnet_cc(fusion_feat_raw)

            # logit_cv1 = F.normalize(logit_cv1, dim=-1)
            # logit_cv2 = F.normalize(logit_cv2, dim=-1) 

            fusion_vs = self.cross_att2(logit_cv1, logit_vs) 
            fusion_cc = self.cross_att1(logit_cv1, logit_cc) 
        else: 
            fusion_cc = logit_cc 
            fusion_vs = logit_vs 
            
        sevty_out = self.fc_severity(fusion_vs) #(bs,4)
        depat_out = self.fc_department(fusion_cc) #(bs,9)
        # ************************************************************************************
        #                            loss  logit_cv
        # ************************************************************************************
        if 'ime' in self.current_task:# Information Mutual Exclusion Loss  
            seu_loss  = objectives.seu_loss4(logit_vs, logit_cc, logit_cv1, logit_cv1)        
            ret.update({'seu_loss': seu_loss})
            
        if 'seu' in self.current_task:# Focal loss    
            seu_loss1  = objectives.compute_itc(logit_vs, logit_cv1, label1)        
            seu_loss2  = objectives.compute_itc(logit_cc, logit_cv1, label2)        
            ret.update({'seu_loss': seu_loss1+seu_loss2})
        
        if 'seu3' in self.current_task:
            tcl_loss1 = objectives.seu_loss(logit_vs, label1)
            tcl_loss2 = objectives.seu_loss(logit_cc, label2)
            ret.update({'seu_loss': tcl_loss1+tcl_loss2})              

        if 'seu2' in self.current_task:# Focal loss    
            seu_loss  = objectives.seu_loss2(sevty_out, depat_out, 
                                            label1, label2, 
                                            self.output_dim_s, self.output_dim_d,
                                            self.per_cls_vs, self.per_cls_cc)        
            ret.update({'seu_loss': seu_loss})

        loss_s = self.CEL(sevty_out, label1)
        loss_d = self.CEL(depat_out, label2)          
        ret.update({'cel_loss': loss_s + loss_d})     

        # ************************************************************************************
        #                            
        # ************************************************************************************
        correct_s = (sevty_out.argmax(dim=1) == label1).sum().item() 
        correct_d = (depat_out.argmax(dim=1) == label2).sum().item() 
        ret.update({'sevty_out': sevty_out, 
                    'depat_out': depat_out, 
                    'correct_s': correct_s, 
                    'correct_d': correct_d})
        return ret, sevty_out, depat_out

class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=1) 
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1) 

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        residual = x
        x = self.res_block(x)
        x += residual
        x = self.global_pool(x)
        x = x.squeeze(-1) 
        return x


import torchvision.models as models

class ResNet1D(nn.Module):
    """
    将一维特征 input_dim=(B, D_in) 
    映射到 output_dim=(B, D_out) 
    基于 torchvision.models.resnetXX。
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 base_model: str = 'resnet18',
                 pretrained: bool = True):
        super().__init__()
        # 1) 加载 backbone
        assert hasattr(models, base_model), f"No such model {base_model}"
        self.backbone: nn.Module = getattr(models, base_model)(pretrained=pretrained)
        
        # 2) 把第一层 conv1 从 in_channels=3 改为 in_channels=1
        #    并把 kernel_size=(7,7) 改成 (1, min(7, input_dim))
        k = 3 #min(3, input_dim)
        pad = k // 2
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=(1, k),
            stride=(1, 2),
            padding=(0, pad),
            bias=False
        )
        # 3) 把最后的全连接层改成输出 output_dim
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D_in)
        return: (B, D_out)
        """
        B, D = x.shape
        # 当作“灰度图”一行： (B, 1, 1, D)
        x = x.view(B, 1, 1, D)
        # 走标准 ResNet 流水线
        return self.backbone(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):  # x shape: (B, 776)
        return self.fc(x)

class TextCNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes=(2, 3, 4), num_filters=256, dilation=2):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=ks)#, dilation=dilation
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = nn.ReLU()(conv_out)
            conv_out = nn.functional.max_pool1d(conv_out, kernel_size=conv_out.shape[2])
            conv_outputs.append(conv_out.squeeze(2))
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
        self.activation = nn.ReLU()
        self.downsample = nn.Conv1d(input_dim, num_filters, kernel_size=1) if input_dim != num_filters else None

    def forward(self, x):
        residual = x
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.activation(out + residual)

class TextResNet(nn.Module):
    def __init__(self, emb_dim, num_classes, kernel_sizes=(2, 3, 4), num_filters=256, dilation=2):
        super(TextResNet, self).__init__()
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(emb_dim, num_filters, kernel_size=ks, dilation=dilation)
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1) 
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        block_outputs = []
        for block in self.residual_blocks:
            block_out = block(x)
            block_out = F.max_pool1d(block_out, kernel_size=block_out.shape[2])
            block_outputs.append(block_out.squeeze(2))
        x = torch.cat(block_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x


from itertools import combinations
class TextResNet8(nn.Module):
    """
    多尺度特征交互网络模型，增强CLS向量的多级语义交互能力，提升分类性能。

    输入:
        x: Tensor of shape (batch_size, emb_dim)  # BERT的CLS向量
    输出:
        logits: Tensor of shape (batch_size, num_classes)
    """
    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        scale_dims=(512, 256, 128),  # 各尺度分支的中间维度
        inter_dim: int = 64,         # 交互向量维度
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_scales = len(scale_dims)
        # 多尺度分支: 每个分支先映射到 scale_dim，然后再投到交互空间 inter_dim
        self.branches = nn.ModuleList()
        for sd in scale_dims:
            self.branches.append(
                nn.Sequential(
                    nn.Linear(emb_dim, sd),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(sd, inter_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        # 最终分类器: 输入维度 = num_scales*inter_dim + C(num_scales,2)*inter_dim
        combs = list(combinations(range(self.num_scales), 2))
        fuse_dim = self.num_scales * inter_dim + len(combs) * inter_dim
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, emb_dim)
        fs = [branch(x) for branch in self.branches]  # 每个 f_i: (batch, inter_dim)
        # 交互: 各尺度对的元素级乘
        inters = []
        for i, j in combinations(range(self.num_scales), 2):
            inters.append(fs[i] * fs[j])  # (batch, inter_dim)
        # 融合所有尺度特征及交互特征
        fuse = torch.cat(fs + inters, dim=1)  # (batch, fuse_dim)
        logits = self.classifier(fuse)
        return logits

class SqueezeExcitation1D(nn.Module):
    """Channel-wise attention via Squeeze-and-Excitation"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, L)
        # Squeeze
        s = x.mean(dim=2)  # (B, C)
        s = self.activation(self.fc1(s))
        s = self.gate(self.fc2(s))  # (B, C)
        # Excitation
        return x * s.unsqueeze(2)

class MultiScaleResBlock(nn.Module):
    """Residual block with multiple kernels and SE attention"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(3,5,7), dilation=1):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = dilation * (k - 1) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, k, dilation=dilation, padding=pad),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
        self.conv1x1 = nn.Conv1d(len(kernel_sizes) * out_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation1D(out_channels)
        self.down = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        res = self.down(x)
        outs = [b(x) for b in self.branches]
        x = torch.cat(outs, dim=1)
        x = self.act(self.bn(self.conv1x1(x)))
        x = self.se(x)
        return self.act(x + res)

class SelfAttention1D(nn.Module):
    """Lightweight multi-head self-attention over sequence"""
    def __init__(self, channels, heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout)

    def forward(self, x):  # x: (B, C, L)
        # prepare for attention: (L, B, C)
        y = x.permute(2, 0, 1)
        y = self.norm(y)
        y, _ = self.attn(y, y, y)
        y = y.permute(1, 2, 0)
        return y + x

class TextResNet2(nn.Module):
    """Hybrid CNN + Self-Attention model for text features"""
    def __init__(self,
                 input_dim,
                 output_dim,
                 channels=128,
                 kernel_sizes=(3,5,7),
                 num_blocks=3,
                 dilation=1,
                 heads=4,
                 dropout=0.2):
        super().__init__()
        self.embed_proj = nn.Conv1d(input_dim, channels, 1)
        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            self.layers.append(
                MultiScaleResBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation*(i+1)
                )
            )
        self.attn = SelfAttention1D(channels, heads=heads)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels//2, output_dim)
        )

    def forward(self, x):  # x: (B, T, F)
        # transpose to (B, F, T)
        x = x.transpose(1, 2)
        x = self.embed_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.attn(x)
        x = self.pool(x).squeeze(2)
        return self.fc(x)