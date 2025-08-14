import torch
from torch.utils.data.dataloader import default_collate
from module.MisData import DataImputer#, DataImputer2
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pkuseg
from module.DataEncoder import VitalSigDataset, ChiefCompDataset
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
import torch.nn.functional as F
vsEncoder = VitalSigDataset()


def cosine_filter_tail_augment(
    vs_aug: torch.Tensor,        # (B, D1)
    cc_aug: torch.Tensor,        # (B, D2)
    lvl_aug: torch.Tensor,       # (B,)
    dept_aug: torch.Tensor,      # (B,)
    all_vs: torch.Tensor,        # (N, D1)
    all_cc: torch.Tensor,        # (N, D2)
    all_lvl: torch.Tensor,       # (N,)
    all_dept: torch.Tensor,      # (N,)
    lvl_thresh: float = 0.2,
    dept_thresh: float = 0.2,
    mode: str = 'vs'             # 'vs' or 'cc'
):
    """
    双路余弦筛选：根据 mode 自动切换 VS/CC 通道。
    mode='vs': 对 vs_aug 与 all_vs 筛选，用 lvl_aug/all_lvl；
    mode='cc': 对 cc_aug 与 all_cc 筛选，用 dept_aug/all_dept。
    返回 (vs_filt, cc_filt, lvl_filt, dept_filt)
    """
    def batch_cosine_filter(aug_tensor, aug_label, full_tensor, full_label, thresh):
        aug_norm  = F.normalize(aug_tensor, dim=1)
        full_norm = F.normalize(full_tensor, dim=1)
        neq_mask = (aug_label.view(-1,1) != full_label.view(1,-1))
        sim = aug_norm @ full_norm.T
        sim = sim.masked_fill(~neq_mask.to(sim.device), -1.0)
        max_sim, _ = sim.max(dim=1)
        keep = max_sim < thresh
        idx  = keep.nonzero(as_tuple=True)[0]
        return idx, keep

    # 根据 mode 切换通道
    if mode == 'vs':
        # vs 路
        idx_vs, mask_vs = batch_cosine_filter(vs_aug, lvl_aug, all_vs, all_lvl, lvl_thresh)
        idx_cc, mask_cc = batch_cosine_filter(cc_aug.view(cc_aug.size(0), -1), dept_aug, \
                                               all_cc.view(all_cc.size(0), -1), all_dept, dept_thresh)
    else:
        # cc 路：swap arguments
        idx_vs, mask_vs = batch_cosine_filter(vs_aug, lvl_aug, all_vs, all_lvl, lvl_thresh)
        idx_cc, mask_cc = batch_cosine_filter(cc_aug.view(cc_aug.size(0), -1), dept_aug, \
                                               all_cc.view(all_cc.size(0), -1), all_dept, dept_thresh)
    # 合并两路 mask
    final_mask = mask_vs & mask_cc
    print(f"[Cosine Filter] VS 保留率={mask_vs.float().mean():.2%}, CC 保留率={mask_cc.float().mean():.2%}, 交集={final_mask.float().mean():.2%}")
    idx_final = final_mask.nonzero(as_tuple=True)[0]
    return vs_aug[idx_final], cc_aug[idx_final], lvl_aug[idx_final], dept_aug[idx_final]

def targeted_augment(
    all_feats: torch.Tensor,
    all_other_feats: torch.Tensor,
    all_labels: torch.Tensor,
    all_other_labels: torch.Tensor,
    target_classes,
    augment_rate: float,
    mode: str = 'vs',              # 'vs' 对 VS 增强；'cc' 对 CC 增强
    sampler: str = 'adasyn',       # 'adasyn' 或 'smote'
    n_neighbors: int = 5,
    smote_kind: str = 'borderline-1'
):
    """
    通用的、有针对性的过采样函数。

    Returns:
        vs_aug, cc_aug, lvl_aug, dept_aug  或 None
    """
    device = all_feats.device
    X = all_feats.cpu().numpy()
    y = all_labels.cpu().numpy()

    # 原始和新增的数量
    orig_counts = {c: int((y == c).sum()) for c in target_classes}
    gen_counts  = {c: int(orig_counts[c] * augment_rate) for c in target_classes}
    # 去掉增量为 0 的
    gen_counts = {c: g for c, g in gen_counts.items() if g > 0}
    if not gen_counts:
        return None

    # 选择采样器并构造 sampling_strategy
    if sampler == 'adasyn':
        strategy = gen_counts  # 新增数
        sampler_obj = ADASYN(
            sampling_strategy=strategy,
            n_neighbors=n_neighbors,
            random_state=42
        )
    else:
        # SMOTE 需要「采样后」的总数 = 原有 + 新增
        total_counts = {c: orig_counts[c] + gen_counts[c] for c in gen_counts}
        sampler_obj = BorderlineSMOTE(
            sampling_strategy=total_counts,
            k_neighbors=n_neighbors,
            kind=smote_kind,
            random_state=42
        )

    # 执行重采样
    X_res, y_res = sampler_obj.fit_resample(X, y)

    # 取出新样本数
    if sampler == 'adasyn':
        n_new = sum(gen_counts.values())
    else:
        # SMOTE 增量 = 新总数 - 原有数
        new_counts = {c: int((y_res == c).sum()) for c in gen_counts}
        n_new = sum(new_counts[c] - orig_counts[c] for c in gen_counts)

    if n_new <= 0:
        return None

    X_new = X_res[-n_new:]
    y_new = y_res[-n_new:]

    # 对齐另一条特征：随机从那些目标类中抽取
    idx_pool = np.where(np.isin(y, list(gen_counts.keys())))[0]
    perm = np.random.choice(idx_pool, size=n_new, replace=True)

    # 返回拼好的增强样本
    if mode == 'vs':
        vs_aug   = torch.from_numpy(X_new).to(device)
        cc_aug   = all_other_feats[perm]
        lvl_aug  = torch.from_numpy(y_new).to(device)
        dept_aug = all_other_labels[perm]
    else:
        vs_aug   = all_feats[perm]
        cc_aug   = torch.from_numpy(X_new).to(device)
        lvl_aug  = all_labels[perm]
        dept_aug = torch.from_numpy(y_new).to(device)

    return vs_aug, cc_aug, lvl_aug, dept_aug

def borderline_smote_cc_tail(all_vs, all_cc, all_lvl, all_dept,
                             tail_depts: set,
                             alpha_cc=0.5,
                             k_neighbors=5,
                             kind='borderline-1'):
    """
    对尾部 Dept 做 Borderline‑SMOTE 增强。
    在全量数据上调用，以避免单一类别的问题。

    Returns:
        vs_aug, cc_aug, lvl_aug, dept_aug 或 None
    """
    device = all_vs.device

    # 扁平化 CC
    cc_tensor = all_cc
    orig_shape = None
    if all_cc.dim() > 2:
        orig_shape = all_cc.shape[1:]
        cc_tensor = all_cc.view(all_cc.size(0), -1)
    X_cc = cc_tensor.cpu().numpy()

    y = all_dept.cpu().numpy()

    # 计算每个 tail dept 需要生成的新样本数量
    counts = {d: int((y == d).sum() * alpha_cc) for d in tail_depts}
    if all(v == 0 for v in counts.values()):
        return None

    sm = BorderlineSMOTE(sampling_strategy=counts,
                         k_neighbors=k_neighbors,
                         kind=kind,
                         random_state=42)
    X_res, y_res = sm.fit_resample(X_cc, y)

    orig_tail_count = sum((y == d).sum() for d in tail_depts)
    res_tail_count  = sum((y_res == d).sum() for d in tail_depts)
    n_new = int(res_tail_count - orig_tail_count)
    if n_new <= 0:
        return None

    X_new = X_res[-n_new:]
    y_new = y_res[-n_new:]

    # 随机对齐 VS/Level
    idx_tail = np.where(np.isin(y, list(tail_depts)))[0]
    perm = np.random.choice(idx_tail, size=n_new, replace=True)

    # 恢复 CC 形状
    cc_new = torch.from_numpy(X_new).float()
    if orig_shape:
        cc_new = cc_new.view(-1, *orig_shape)
    cc_new = cc_new.to(device)

    vs_aug   = all_vs[perm]
    cc_aug   = cc_new
    lvl_aug  = all_lvl[perm]
    dept_aug = torch.from_numpy(y_new).to(device)

    return vs_aug, cc_aug, lvl_aug, dept_aug

def build_sampler(dataset, label_key='Dept_digit', beta=1.0, alpha=0.5):
    labels = [sample[label_key] for sample in dataset]
    cnt = Counter(labels)
    max_cnt = max(cnt.values())
    class_weights = {c: (cnt[c] / max_cnt)**beta for c in cnt}
    sample_weights = [class_weights[l] for l in labels]
    num_samples = int(len(dataset) * (1 + alpha))
    return WeightedRandomSampler(sample_weights, num_samples, replacement=True)

def collate_fn(batch):
    max_len_cc = max(s['CC_tokens'].size(0) for s in batch)
    for s in batch:
        pad = max_len_cc - s['CC_tokens'].size(0)
        if pad > 0:
            zeros = torch.zeros(pad, s['CC_tokens'].size(1), device=s['CC_tokens'].device)
            s['CC_tokens'] = torch.cat([s['CC_tokens'], zeros], dim=0)
    return default_collate(batch)

def Match(indices, JointFeature):
    return [(JointFeature['VS'][idx], 
             JointFeature['CC'][idx],
             JointFeature['Level'][idx],
             JointFeature['Depart'][idx],
             ) for idx in indices]

def Data_Indices(args):
    stopwords = pd.read_csv(args.stopword_path, quoting=3, sep="\t", encoding='utf-8')
    segcut = pkuseg.pkuseg(model_name="medicine", user_dict="default", postag=False)
    column = ['性别','出生日期','到院方式','分诊印象',
              'T℃','P(次/分)','R(次/分)','BP(mmHg)','SpO2',
              '级别','去向']
    df_dirty_raw = pd.read_csv(args.data_path_dirty, engine='python')
    df_dirty = pd.DataFrame(df_dirty_raw, columns=column)
    df_dirty['__split__'] = 'dirty'
    if args.data_quality == 'clean':
        df_clean_raw = pd.read_csv(args.data_path_clean, engine='python')
        df_clean = pd.DataFrame(df_clean_raw, columns=column)
        df_clean['__split__'] = 'clean'
        raw_all = pd.concat([df_dirty, df_clean], ignore_index=True)
    else:
        raw_all = df_dirty.copy()

    if args.data_quality == 'clean':
        fit_df = raw_all
        trans_df = raw_all[raw_all['__split__']=='clean'].reset_index(drop=True)
    else:
        fit_df = raw_all[raw_all['__split__']=='dirty']
        trans_df = fit_df.reset_index(drop=True)
    
    if args.ImputMode != '':
        imputer = DataImputer(args, latent_dim=16, learning_rate=1e-3, epochs=2000)
        imputed_feats = imputer.impute(trans_df)
        imputed = imputed_feats.copy()
        imputed['分诊印象'] = trans_df['分诊印象'].values
        imputed['级别']     = trans_df['级别'].values
        imputed['去向']     = trans_df['去向'].values
        train_set, valid_set, test_set, dic1, dic2, Y1, Y2 = DatasetProce(args, imputed, column, stopwords, segcut)
    else:
        train_set, valid_set, test_set, dic1, dic2, Y1, Y2 = DatasetProce(args, trans_df, column, stopwords, segcut)

    # 6. 最后构造 DataLoader
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)

    inverse_dic1 = {v: SeverityLib[k] for k, v in dic1.items()}
    inverse_dic2 = {v: DepartmentLib[k] for k, v in dic2.items()}

    save_path = os.path.join(args.cache_dir, 'inverse_dicts.pth')
    torch.save({
        'inverse_dic1': inverse_dic1,
        'inverse_dic2': inverse_dic2
    }, save_path)
    return train_loader, valid_loader, test_loader, Y1, Y2, inverse_dic1, inverse_dic2
    

def DatasetProce(args, data_proce, column, stopwords, segcut):    
    data = data_proce[column].iloc[:args.length].reset_index(drop=True)
    ChiefComp = data['分诊印象'][:args.length].fillna("").astype(str)
    unique_levels = data['级别'][:args.length].dropna().unique()
    unique_departments = data['去向'][:args.length].dropna().unique()
    dic1 = {level: idx for idx, level in enumerate(unique_levels)}
    dic2 = {dept: idx for idx, dept in enumerate(unique_departments)}
    Y1 = torch.tensor(data[:args.length]['级别'].map(dic1).fillna(-1).astype(int).values).long()
    Y2 = torch.tensor(data[:args.length]['去向'].map(dic2).fillna(-1).astype(int).values).long()

    exclude = [w[0] for w in np.asarray(stopwords)] + [' ']
    def langseg(text, exclude):
        words = segcut.cut(text)
        return " ".join([w for w in words if w not in exclude])

    im = ChiefComp.apply(lambda x: langseg(x, exclude)).tolist()

    vs = torch.cat(vsEncoder.Structure(data), dim=1)
    if args.SFD:
        vs = vsEncoder.SFD_encoder(vs)
    # 划分 train/valid/test
    train_idx, vt_idx = train_test_split(np.arange(len(Y1)), test_size=0.2, random_state=args.rand_seed)
    val_idx, test_idx = train_test_split(vt_idx, test_size=0.5, random_state=args.rand_seed)

    JointFeature = {'VS': vs[:args.length],'CC': im[:args.length],'Level': Y1[:args.length], 'Depart':Y2[:args.length]}    
    # 构造 Dataset 与堆栈所有 train 样本
    train_data = Match(train_idx, JointFeature)
    valid_data = Match(val_idx, JointFeature)
    train_set  = ChiefCompDataset(args, train_data, str(args.rand_seed) + '_'+ args.data_quality+'_'+args.ImputMode + '_train')
    valid_set  = ChiefCompDataset(args, valid_data, str(args.rand_seed) + '_'+ args.data_quality+'_'+args.ImputMode + '_valid')
    
    test_data = Match(test_idx, JointFeature)
    test_set = ChiefCompDataset(args, test_data,    str(args.rand_seed) + '_'+ args.data_quality+'_'+args.ImputMode + '_test')  
    
    return train_set, valid_set, test_set, dic1, dic2, Y1, Y2

SeverityLib={
    '一级':'Level 1', #AP=0.2760, N=46
    '二级':'Level 2', #AP=0.3005, N=221
    '三级':'Level 3', #AP=0.7250, N=3784
    '四级':'Level 4', #AP=0.9747, N=12069
             }
DepartmentLib={
    '内科':'Internal Medicine',       #AP=0.9621, N=5525
    '产科':'Obstetrics',              #AP=0.9900, N=540
    '外科':'Surgery',                 #AP=0.9909, N=7743
    '眼科':'Ophthalmology',           #AP=0.9882, N=263
    '妇科':'Gynecology',              #AP=0.7535, N=226
    '耳鼻喉':'Otolaryngology',        #AP=0.9476, N=573
    '神经外科':'Neurosurgery',        #AP=0.0035, N=7
    '创伤救治中心':'Trauma Center',   #AP=0.0420, N=5
    '抢救室':'Rescuration Room',
    '骨科':'Orthopedics',             #AP=0.1328, N=55
    '神经内科':'Neurology',           #AP=0.7684, N=1183
              }   