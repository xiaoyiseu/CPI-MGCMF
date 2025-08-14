import torch
import torch.nn as nn
import torch.nn.functional as F

def _compute_class_weights(samples_per_cls, beta, device=None):
    """
    Compute Class Balanced weights based on effective number of samples.
    """
    counts = torch.tensor(samples_per_cls, dtype=torch.float, device=device)
    effective_num = 1.0 - torch.pow(beta, counts)
    class_weights = (1.0 - beta) / effective_num
    class_weights = class_weights / class_weights.sum() * counts.numel()
    return class_weights

class ClassBalancedFocalLoss(nn.Module):
    """
    Combined Class-Balanced and Focal Loss for multi-class tasks.
    Features:
      - pure torch implementation on GPU
      - optional label smoothing
      - Softmax-based focal variant for stability
      - precomputed class weights as buffer
    """
    def __init__(self,
                 samples_per_cls,
                 num_classes,
                 beta=0.9999,
                 gamma=2.0,
                 smooth_eps=0.0,
                 reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.smooth_eps = smooth_eps
        self.reduction = reduction

        # precompute class weights once
        class_weights = _compute_class_weights(samples_per_cls, beta)
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, labels):
        device = logits.device
        # one-hot encode and optional smoothing
        labels_onehot = F.one_hot(labels, self.num_classes).float().to(device)
        if self.smooth_eps > 0:
            labels_onehot = labels_onehot * (1 - self.smooth_eps) + self.smooth_eps / self.num_classes

        # log-probabilities and probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # focal factor (softmax-based)
        focal_factor = torch.pow(1.0 - probs, self.gamma)

        # sample-wise class weight extraction
        weight_per_sample = self.class_weights[labels].unsqueeze(1)

        # compute loss matrix and reduce
        loss_matrix = - weight_per_sample * focal_factor * labels_onehot * log_probs
        loss = loss_matrix.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class MultiTaskCBFocal(nn.Module):
    """
    Multi-task wrapper combining severity and department losses.
    """
    def __init__(self,
                 vs_samples_per_cls,
                 cc_samples_per_cls,
                 vs_num_classes,
                 cc_num_classes,
                 beta=0.9999,
                 gamma=2.0,
                 smooth_eps=0.0,
                 weights=(1.0, 1.0)):
        super().__init__()
        self.loss_vs = ClassBalancedFocalLoss(
            samples_per_cls=vs_samples_per_cls,
            num_classes=vs_num_classes,
            beta=beta, gamma=gamma,
            smooth_eps=smooth_eps,
            reduction='mean')
        self.loss_cc = ClassBalancedFocalLoss(
            samples_per_cls=cc_samples_per_cls,
            num_classes=cc_num_classes,
            beta=beta, gamma=gamma,
            smooth_eps=smooth_eps,
            reduction='mean')
        self.w_vs, self.w_cc = weights

    def forward(self, logit_vs, logit_cc, label_vs, label_cc):
        loss1 = self.loss_vs(logit_vs, label_vs)
        loss2 = self.loss_cc(logit_cc, label_cc)
        return self.w_vs * loss1 + self.w_cc * loss2

def seu_loss2(logit_vs, logit_cc,
             label1, label2,
             output_dim_s, output_dim_d,
             per_cls_vs, per_cls_cc,
             beta=0.6, gamma=2.0,
             smooth_eps=1e-4,
             task_weights=(1.0, 1.0)):
    """
    Compute multi-task Class-Balanced Focal Loss.

    Args:
      logit_vs: [B, output_dim_s] logits for severity task
      logit_cc: [B, output_dim_d] logits for department task
      label1:   [B] targets for severity (int)
      label2:   [B] targets for department (int)
      output_dim_s: number of severity classes
      output_dim_d: number of department classes
      per_cls_vs: list of sample counts per severity class
      per_cls_cc: list of sample counts per department class
      beta, gamma: hyperparams for CB and focal
      smooth_eps: optional label smoothing epsilon
      task_weights: tuple of two floats for multi-task weighting

    Returns:
      combined loss scalar
    """
    # instantiate multi-task criterion
    criterion = MultiTaskCBFocal(
        vs_samples_per_cls=per_cls_vs,
        cc_samples_per_cls=per_cls_cc,
        vs_num_classes=output_dim_s,
        cc_num_classes=output_dim_d,
        beta=beta, gamma=gamma,
        smooth_eps=smooth_eps,
        weights=task_weights
    ).to(logit_vs.device)

    # compute and return
    return criterion(logit_vs, logit_cc, label1.to(logit_vs.device), label2.to(logit_vs.device))

#***************************************************
#               taylor_ce_loss
#***************************************************
# def taylor_ce_loss(outputs, labels, series):
def seu_loss3(outputs, labels, series):
    device = outputs.device
    dtype  = outputs.dtype
    n      = series
    k      = outputs.size(1)

    # 1) compute p = probability assigned to the true class
    sm_outputs    = F.softmax(outputs, dim=1)
    one_hot       = F.one_hot(labels, k).float().to(device)
    final_outputs = (sm_outputs * one_hot).sum(dim=1)     # shape: [batch]

    # 2) build the Taylor series term by term
    total = torch.zeros_like(final_outputs, device=device, dtype=dtype)
    for i in range(n):  # i = 0 .. n-1
        sign   = torch.tensor((-1.0) ** (i+1),
                              device=device,
                              dtype=dtype)
        power  = torch.pow(final_outputs - 1, i+1)
        total += sign * power / (i+1)

    return total.mean()



def seu_loss(
    logit_cc: torch.Tensor,
    logit_vs: torch.Tensor,
    logit_cv: torch.Tensor,
    tau: float = 0.01,
    alpha_js: float = 5.0,
    beta_frob: float = 1.0,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    计算 CC–CV 与 VS–CV 相似度矩阵差异的正值 score：
      score = α * JS_mean + β * Frobenius_diff

    返回 ≥0。

    参数：
      logit_cc, logit_vs, logit_cv: (B, D)
      tau: softmax 温度
      alpha_js: 行级 JS 项权重
      beta_frob: Frobenius 项权重
    """
    B = logit_cc.size(0)

    # 1) 归一化
    cc_n = F.normalize(logit_cc, dim=1)
    vs_n = F.normalize(logit_vs, dim=1)
    cv_n = F.normalize(logit_cv, dim=1)

    # 2) 相似度矩阵
    S_cc = cc_n @ cv_n.t()  # (B, B)
    S_vs = vs_n @ cv_n.t()  # (B, B)

    # 3) 行 softmax → 分布
    P = F.softmax(S_cc / tau, dim=1).clamp(min=epsilon)
    Q = F.softmax(S_vs / tau, dim=1).clamp(min=epsilon)
    M = 0.5 * (P + Q)

    # 4) 行级 JS 散度
    kl_pm = (P * (P.log() - M.log())).sum(dim=1)
    kl_qm = (Q * (Q.log() - M.log())).sum(dim=1)
    js_vals = 0.5 * (kl_pm + kl_qm)      # (B,)
    js_mean = js_vals.mean()            # ≥0

    # 5) Frobenius 差异
    diff = S_cc - S_vs
    frob = diff.pow(2).sum().sqrt() / (B * B)  # ≥0

    # 6) 组合 score
    return alpha_js * js_mean + beta_frob * frob




def seu_loss4(
    logit_cc: torch.Tensor,
    logit_vs: torch.Tensor,
    logit_cv: torch.Tensor,
    tau=0.01,
    epsilon=1e-8,
    n_iter: int = 50
) -> torch.Tensor:
    B = logit_cc.size(0)
    device = logit_cc.device
    dtype  = logit_cc.dtype

    # L2 normalize
    cc_n = F.normalize(logit_cc, dim=1)
    vs_n = F.normalize(logit_vs, dim=1)
    cv_n = F.normalize(logit_cv, dim=1)

    # similarity matrices
    S_cc = cc_n @ cv_n.t() / tau
    S_vs = vs_n @ cv_n.t() / tau

    # row‐softmax → distributions
    P = F.softmax(S_cc, dim=1).clamp(min=epsilon)
    Q = F.softmax(S_vs, dim=1).clamp(min=epsilon)

    # 归一化索引 [0,1]
    idx = torch.linspace(0, 1, steps=B, device=device, dtype=dtype)
    C = (idx.view(-1,1) - idx.view(1,-1)).pow(2)  # now in [0,1]

    # Sinkhorn kernel
    K = torch.exp(-C / epsilon).clamp(min=epsilon)

    # Sinkhorn iterations
    u = torch.ones_like(P) / B
    v = torch.ones_like(Q) / B
    for _ in range(n_iter):
        u = P / (K @ v.t()).t()
        v = Q / (K.t() @ u.t()).t()

    # transport tensor & distance
    u_exp = u.unsqueeze(2)
    v_exp = v.unsqueeze(1)
    T = u_exp * K.unsqueeze(0) * v_exp
    sink = (T * C.unsqueeze(0)).sum(dim=(1,2))

    return sink.mean()
