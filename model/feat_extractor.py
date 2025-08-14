import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True)  # LSTM层
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全连接层
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x) 
        output = self.fc(lstm_out[:, -1, :]) 
        return output

class Encoder(nn.Module):
    def __init__(self, hidden_dim = 64, output_dim=768):
        super(Encoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_vs = FeatureExtractor(input_dim=28, 
                                           hidden_dim=hidden_dim, 
                                           output_dim=output_dim).to(self.device)
        self.encoder_cc = FeatureExtractor(input_dim=768, 
                                           hidden_dim=hidden_dim, 
                                           output_dim=output_dim).to(self.device)
    def CasualWeighted(self, vs_data, labels):
        num_classes = labels.unique().numel()  # 标签类别数
        n_feat = vs_data.shape[1]  # 生命体征特征数
        prob_matrix = torch.zeros_like(vs_data, dtype=torch.float)
        for label in range(num_classes):
            vs_subset = vs_data[labels == label]
            for col in range(n_feat):
                unique_vals, counts = vs_subset[:, col].unique(return_counts=True)
                probs = counts.float() / vs_subset.shape[0]
                for val, prob in zip(unique_vals, probs):
                    mask = (vs_data[:, col] == val) & (labels == label)
                    prob_matrix[mask, col] = prob
        return vs_data * prob_matrix

    def extract_features(self, batch):
        """提取并处理特征的函数"""
        VitalSign = batch['VS']
        ChiefComplaint = batch['CC']
        label1, label2 = batch['Level'], batch['Depart']

        cc_feat = ChiefComplaint.squeeze()
        vs_feat_cw = self.CasualWeighted(VitalSign, label1)
        vs_feat = self.encoder_vs(vs_feat_cw) 
        # cc_feat0 = self.encoder_cc(cc_feat) 
        return cc_feat, vs_feat, label1, label2