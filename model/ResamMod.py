import torch
from typing import Callable, Optional

# class TailResampler(torch.nn.Module):
#     def __init__(self,
#                  alpha_cc: float = 0.5,
#                  alpha_vs: float = 0.5,
#                  vs_transform: Optional[Callable] = None,
#                  cc_transform: Optional[Callable] = None):
#         """
#         :param alpha_cc: CC重采样比例
#         :param alpha_vs: VS重采样比例
#         :param vs_transform: 对新增 VS 样本做数据增强
#         :param cc_transform: 对新增 CC 样本做数据增强
#         """
#         super().__init__()
#         self.alpha_cc = alpha_cc
#         self.alpha_vs = alpha_vs
#         self.vs_transform = vs_transform
#         self.cc_transform = cc_transform

#     def _resample(self, src_tensor, target_tensor, label, alpha, transform_src=None, transform_target=None):
#         B = src_tensor.size(0)
#         counts = torch.bincount(label, minlength=int(label.max().item()) + 1).float()
#         sample_probs = counts[label] / counts.sum()
#         n_new = int(B * alpha)
#         idx_new = torch.multinomial(sample_probs, n_new, replacement=True)

#         src_new = src_tensor[idx_new]
#         target_new = target_tensor[idx_new]
#         label_new = label[idx_new]

#         if transform_src:
#             src_new = transform_src(src_new)
#         if transform_target:
#             target_new = transform_target(target_new)

#         return src_new, target_new, label_new

#     def forward(self,
#                 encoder_vs: torch.Tensor,
#                 enc_cc_cls: torch.Tensor,
#                 label1: torch.Tensor,
#                 label2: torch.Tensor):
#         # 原始样本
#         vs_all = [encoder_vs]
#         cc_all = [enc_cc_cls]
#         lvl_all = [label1]
#         dept_all = [label2]

#         # 模式1：固定VS，采样CC
#         cc_new, vs_dup1, dept_new = self._resample(
#             src_tensor=enc_cc_cls, target_tensor=encoder_vs, label=label2,
#             alpha=self.alpha_cc, transform_src=self.cc_transform)

#         lvl_dup1 = label1[:vs_dup1.size(0)]  # VS未变，label1维持与目标一致

#         # 模式2：固定CC，采样VS
#         vs_new, cc_dup2, lvl_new = self._resample(
#             src_tensor=encoder_vs, target_tensor=enc_cc_cls, label=label1,
#             alpha=self.alpha_vs, transform_src=self.vs_transform)

#         dept_dup2 = label2[:cc_dup2.size(0)]  # CC未变，label2维持与目标一致

#         # 拼接全部数据
#         vs_all += [vs_dup1, vs_new]
#         cc_all += [cc_new, cc_dup2]
#         lvl_all += [lvl_dup1, lvl_new]
#         dept_all += [dept_new, dept_dup2]

#         vs_ext = torch.cat(vs_all, dim=0)
#         cc_ext = torch.cat(cc_all, dim=0)
#         lvl_ext = torch.cat(lvl_all, dim=0)
#         dept_ext = torch.cat(dept_all, dim=0)

#         return vs_ext, cc_ext, lvl_ext, dept_ext


# def compute_centroids(tensor: torch.Tensor, labels: torch.Tensor):
#     """
#     计算每个类别的质心。
#     返回 dict: label -> centroid tensor（与 tensor 维度一致）
#     """
#     centroids = {}
#     for lbl in torch.unique(labels):
#         mask = labels == lbl
#         centroids[int(lbl.item())] = tensor[mask].mean(dim=0)
#     return centroids

# class TailResampler(torch.nn.Module):
#     def __init__(self,
#                  alpha_cc: float = 0.5,
#                  alpha_vs: float = 0.5,
#                  beta: float = 1.0):
#         """
#         同时对 CC 和 VS 进行长尾重采样，并基于质心回归+概率分布做数据增强。

#         :param alpha_cc: 对 CC 重采样比例
#         :param alpha_vs: 对 VS 重采样比例
#         :param beta: 控制噪声幅度（与类分布协方差成比例）
#         """
#         super().__init__()
#         assert alpha_cc >= 0 and alpha_vs >= 0
#         self.alpha_cc = alpha_cc
#         self.alpha_vs = alpha_vs
#         self.beta = beta

#     def _augment(self,
#                  tensor: torch.Tensor,
#                  labels: torch.Tensor,
#                  centroids: dict,
#                  beta: float):
#         """
#         对 tensor 中每个样本做回归到对应质心并加随机噪声。
#         x' = x + gamma*(centroid[label] - x) + noise
#         gamma 从 [0,1] 随机取， noise ~ N(0, cov * beta)
#         这里 cov 简化为 batch 内同类别的方差。
#         """
#         augmented = []
#         unique = torch.unique(labels)
#         # 先计算每类样本协方差对角（简化）
#         var_map = {}
#         for lbl in unique:
#             mask = labels == lbl
#             var_map[int(lbl.item())] = tensor[mask].var(dim=0, unbiased=False)
#         # 对每个样本
#         for x, lbl in zip(tensor, labels):
#             c = centroids[int(lbl.item())]
#             v = var_map[int(lbl.item())]
#             gamma = torch.rand(1, device=x.device).item()
#             noise = torch.randn_like(x) * torch.sqrt(v * beta + 1e-6)
#             x_aug = x + gamma * (c - x) + noise
#             augmented.append(x_aug)
#         return torch.stack(augmented, dim=0)

#     def forward(self,
#                 encoder_vs: torch.Tensor,
#                 enc_cc_cls: torch.Tensor,
#                 label1: torch.Tensor,
#                 label2: torch.Tensor):
#         """
#         返回扩充后的 vs_ext, cc_ext, lvl_ext, dept_ext
#         """
#         B = encoder_vs.size(0)
#         # 计算质心
#         cent_vs = compute_centroids(encoder_vs, label1)
#         cent_cc = compute_centroids(enc_cc_cls, label2)

#         vs_list, cc_list, lvl_list, dept_list = [encoder_vs], [enc_cc_cls], [label1], [label2]

#         # 模式1: 固定VS，采样CC
#         n_cc = int(B * self.alpha_cc)
#         probs_cc = torch.bincount(label2).float()[label2] / label2.size(0)
#         idx_cc = torch.multinomial(probs_cc, n_cc, replacement=True)
#         cc_sel = enc_cc_cls[idx_cc]; vs_sel1 = encoder_vs[idx_cc]
#         cc_aug = self._augment(cc_sel, label2[idx_cc], cent_cc, self.beta)
#         vs_list.append(vs_sel1)
#         cc_list.append(cc_aug)
#         lvl_list.append(label1[idx_cc]); dept_list.append(label2[idx_cc])

#         # 模式2: 固定CC，采样VS
#         n_vs = int(B * self.alpha_vs)
#         probs_vs = torch.bincount(label1).float()[label1] / label1.size(0)
#         idx_vs = torch.multinomial(probs_vs, n_vs, replacement=True)
#         vs_sel2 = encoder_vs[idx_vs]; cc_sel2 = enc_cc_cls[idx_vs]
#         vs_aug = self._augment(vs_sel2, label1[idx_vs], cent_vs, self.beta)
#         vs_list.append(vs_aug)
#         cc_list.append(cc_sel2)
#         lvl_list.append(label1[idx_vs]); dept_list.append(label2[idx_vs])

#         vs_ext = torch.cat(vs_list, dim=0)
#         cc_ext = torch.cat(cc_list, dim=0)
#         lvl_ext = torch.cat(lvl_list, dim=0)
#         dept_ext = torch.cat(dept_list, dim=0)
#         return vs_ext, cc_ext, lvl_ext, dept_ext


import torch
from torch import Tensor
from typing import Tuple

class TailResampler(torch.nn.Module):
    """
    仅对长尾类别进行基于概率分布的重采样增强。
    支持两种采样策略：
      - 'head_dist': 在尾部类别中，按头部优先（多数类更多采样）
      - 'tail_dist': 在尾部类别中，按尾部优先（少数类更多采样）

    采样仅来自尾部类别，头部类别样本保留不变。
    可通过 tail_percentile 调整“头尾”分割门槛。
    """
    def __init__(
        self,
        tail_ratio_cc: float = 1.0,
        tail_ratio_vs: float = 1.0,
        dist_mode: str = 'tail_dist',
        tail_percentile: float = 50.0
    ):
        super().__init__()
        assert dist_mode in ('head_dist', 'tail_dist'), "dist_mode must be 'head_dist' or 'tail_dist'"
        assert tail_ratio_cc >= 0 and tail_ratio_vs >= 0, "tail ratios must be >=0"
        assert 0 <= tail_percentile <= 100, "tail_percentile must be in [0,100]"
        self.tail_ratio_cc = tail_ratio_cc
        self.tail_ratio_vs = tail_ratio_vs
        self.dist_mode = dist_mode
        self.tail_percentile = tail_percentile

    def _split_tail(self, labels: Tensor) -> Tensor:
        # 计算所有类别频次，并基于分位数阈值划分尾部类别
        counts = torch.bincount(labels, minlength=int(labels.max())+1).float()
        thresh = float(torch.quantile(counts, self.tail_percentile / 100.0))
        tail_classes = (counts < thresh).nonzero(as_tuple=False).view(-1)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in tail_classes:
            mask |= (labels == c)
        return mask.nonzero(as_tuple=False).view(-1)

    def _make_probs(self, labels: Tensor, idxs: Tensor) -> Tensor:
        # 在给定尾部 idxs 内，按 head_dist 或 tail_dist 计算采样概率
        sub_labels = labels[idxs]
        counts = torch.bincount(sub_labels, minlength=int(sub_labels.max())+1).float().clamp(min=1e-6)
        if self.dist_mode == 'head_dist':
            probs = counts[sub_labels]
        else:
            probs = (1.0 / counts)[sub_labels]
        return probs / probs.sum()

    def forward(
        self,
        encoder_vs: Tensor,
        enc_cc_cls: Tensor,
        label1: Tensor,
        label2: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B = encoder_vs.size(0)
        device = encoder_vs.device
        # 确保所有输入都在同一设备
        encoder_vs  = encoder_vs.to(device)
        enc_cc_cls  = enc_cc_cls.to(device)
        label1      = label1.to(device)
        label2      = label2.to(device)
        all_idx = torch.arange(B, device=device)

        # CC 分支采样
        tail_idx_cc = self._split_tail(label2)
        head_idx_cc = all_idx[~torch.isin(all_idx, tail_idx_cc)]
        desired_tail_cc = int(head_idx_cc.numel() * self.tail_ratio_cc)
        need_cc = max(0, desired_tail_cc - tail_idx_cc.numel())
        if need_cc > 0 and tail_idx_cc.numel() > 0:
            probs_cc = self._make_probs(label2, tail_idx_cc)
            choice_cc = torch.multinomial(probs_cc, need_cc, replacement=True)
            samp_cc = tail_idx_cc[choice_cc]
            new_cc = enc_cc_cls[samp_cc]
            new_vs_cc = encoder_vs[samp_cc]
            new_lvl_cc = label1[samp_cc]
            new_dep_cc = label2[samp_cc]
        else:
            new_cc = torch.empty((0,)+enc_cc_cls.shape[1:], device=device)
            new_vs_cc = torch.empty((0,)+encoder_vs.shape[1:], device=device)
            new_lvl_cc = torch.empty((0,), dtype=label1.dtype, device=device)
            new_dep_cc = torch.empty((0,), dtype=label2.dtype, device=device)

        # VS 分支采样
        tail_idx_vs = self._split_tail(label1)
        head_idx_vs = all_idx[~torch.isin(all_idx, tail_idx_vs)]
        desired_tail_vs = int(head_idx_vs.numel() * self.tail_ratio_vs)
        need_vs = max(0, desired_tail_vs - tail_idx_vs.numel())
        if need_vs > 0 and tail_idx_vs.numel() > 0:
            probs_vs = self._make_probs(label1, tail_idx_vs)
            choice_vs = torch.multinomial(probs_vs, need_vs, replacement=True)
            samp_vs = tail_idx_vs[choice_vs]
            new_vs = encoder_vs[samp_vs]
            new_cc_vs = enc_cc_cls[samp_vs]
            new_lvl_vs = label1[samp_vs]
            new_dep_vs = label2[samp_vs]
        else:
            new_vs   = torch.empty((0,)+encoder_vs.shape[1:], device=device)
            new_cc_vs= torch.empty((0,)+enc_cc_cls.shape[1:], device=device)
            new_lvl_vs = torch.empty((0,), dtype=label1.dtype, device=device)
            new_dep_vs = torch.empty((0,), dtype=label2.dtype, device=device)

        # 拼接结果，并确保 CC 为 [N,1,D]
        vs_ext   = torch.cat([encoder_vs, new_vs_cc, new_vs], dim=0)
        cc_ext   = torch.cat([enc_cc_cls, new_cc,  new_cc_vs], dim=0)
        lvl_ext  = torch.cat([label1,     new_lvl_cc, new_lvl_vs], dim=0)
        dept_ext = torch.cat([label2,     new_dep_cc, new_dep_vs], dim=0)
        if cc_ext.dim() == 2:
            cc_ext = cc_ext.unsqueeze(1)
        return vs_ext, cc_ext, lvl_ext, dept_ext

# 使用示例
if __name__ == '__main__':
    B, V, C, K1, K2 = 32, 64, 128, 4, 10
    encoder_vs = torch.randn(B, V).cuda()
    enc_cc_cls = torch.randn(B, C).cuda()
    label1     = torch.randint(0, K1, (B,)).cuda()
    label2     = torch.randint(0, K2, (B,)).cuda()

    resampler = TailResampler(
        tail_ratio_cc=1.0, tail_ratio_vs=0.5,
        dist_mode='tail_dist', tail_percentile=30.0
    )
    vs_ext, cc_ext, lvl_ext, dept_ext = resampler(
        encoder_vs, enc_cc_cls, label1, label2
    )
    print(vs_ext.device, cc_ext.device)
    print(vs_ext.shape, cc_ext.shape)

