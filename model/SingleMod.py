import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class)
        )
    def forward(self, x):  # x: [B, D]
        return self.net(x)

class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, rnn_type='lstm', num_layers=1):
        super().__init__()
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        # 我们把整个特征向量当作长度=1的序列来跑
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, num_class)
    def forward(self, x):  # x: [B, D]
        x = x.unsqueeze(1)  # [B, 1, D]
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])  # [B, H] -> [B, num_class]

class TextCNNClassifier(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_sizes, num_class, dropout=0.5):
        """
        input_dim: 特征维度 D
        num_filters: 每个卷积核通道数
        kernel_sizes: list of int, 卷积核宽度（感受野）
        num_class: 输出类别数
        """
        super().__init__()
        # 视作 [B, 1, D] 做 1D 卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(1, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_class)

    def forward(self, x):  # x: [B, D]
        x = x.unsqueeze(1)  # [B, 1, D]
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))       # [B, F, D]
            # 池化到每个通道一个值
            c = nn.functional.adaptive_max_pool1d(c, 1).squeeze(2)  # [B, F]
            conv_outs.append(c)
        feat = torch.cat(conv_outs, dim=1)   # [B, F * len(kernel_sizes)]
        feat = self.dropout(feat)
        return self.fc(feat)                 # [B, num_class]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, nhead, nlayers, num_class, dim_feedforward=512, dropout=0.1):
        """
        input_dim: 特征维度 D
        nhead: 多头注意力头数
        nlayers: TransformerEncoder 层数
        num_class: 输出类别数
        dim_feedforward: 前馈网络维度
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, num_class)

    def forward(self, x):  # x: [B, D]
        # 为了符合 Transformer，需要 [B, seq, d_model]，这里把 D 作为 seq
        x = x.unsqueeze(2)              # [B, D, 1]
        x = x.permute(0, 2, 1)          # [B, 1, D]
        out = self.transformer(x)       # [B, 1, D]
        out = out.permute(0, 2, 1)      # [B, D, 1]
        feat = self.pool(out).squeeze(2)  # [B, D]
        return self.fc(feat)            # [B, num_class]
