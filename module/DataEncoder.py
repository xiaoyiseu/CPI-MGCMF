import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import os
from collections import Counter, defaultdict
# import jieba.posseg as pseg
from module.StructureEncoder import StructureDataEncoder
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
import pkuseg#专业领域分词
import random
# import jieba
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

segcut = pkuseg.pkuseg(model_name = "medicine", user_dict = "default", postag = False)  # 程序会自动下载所对应的细领域模型
# 全局缓存 CLIP 模型，避免多次加载或下载
_clip_model = None
_clip_preprocess = None

def load_clip_once(device, cache_dir):
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        # 确保 cache_dir 存在且为目录
        if os.path.exists(cache_dir) and not os.path.isdir(cache_dir):
            os.remove(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        # 直接加载或下载模型
        _clip_model, _clip_preprocess = load_from_name(
            "ViT-B-16", device=device, download_root=cache_dir
        )
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_preprocess

class VitalSigDataset:
    def __init__(self):
        self.digit = StructureDataEncoder()
        self.num_classes = {
            '到院方式': 4,  
            '性别': 3,
            '出生日期': 5,
            'T℃': 3,
            'P(次/分)': 3,
            'R(次/分)': 3,
            'BP(mmHg)': 5,
            'SpO2': 3
        }        
    def one_hot(self, y, num_classes=None):
        """Convert to one-hot encoding."""
        y_tensor = torch.tensor(y)
        if num_classes is None:
            num_classes = y_tensor.max() + 1
        return torch.nn.functional.one_hot(y_tensor, num_classes=num_classes).float()

    def Structure(self, data):
        ar = self.one_hot(data['到院方式'].apply(lambda x: self.digit.Arr_way(x)).values, self.num_classes['到院方式'])
        g  = self.one_hot(data['性别'].apply(lambda x: self.digit.Gender(x)).values, self.num_classes['性别'])
        a  = self.one_hot(data['出生日期'].apply(lambda x: self.digit.Age(x)).values, self.num_classes['出生日期'])
        t  = self.one_hot(data['T℃'].apply(lambda x: self.digit.Temperature(x)).values, self.num_classes['T℃'])
        p  = self.one_hot(data['P(次/分)'].apply(lambda x: self.digit.Pulse(x)).values, self.num_classes['P(次/分)'])
        r  = self.one_hot(data['R(次/分)'].apply(lambda x: self.digit.Respiration(x)).values, self.num_classes['R(次/分)'])
        bp = self.one_hot(data['BP(mmHg)'].apply(lambda x: self.digit.BloodPressure(x)).values, self.num_classes['BP(mmHg)'])
        s  = self.one_hot(data['SpO2'].apply(lambda x: self.digit.SpO2(x)).values, self.num_classes['SpO2'])
        return ar, g, a, t, p, r, bp, s

    def SFD_encoder(self, vs):#在manerger.py中进行了调用
        batch_size, _ = vs.shape
        indices = vs.nonzero(as_tuple=True)[1].view(batch_size, -1)
        num_indices = indices.shape[1]
        distance_matrix = torch.zeros((batch_size, num_indices, num_indices), dtype=torch.float32)
        for idx in range(batch_size):
            feature_indices = indices[idx].float().view(-1, 1)
            dist_matrix = torch.cdist(feature_indices, feature_indices, p=1)
            distance_matrix[idx] = dist_matrix
        tri_indices = torch.triu_indices(distance_matrix.size(1), distance_matrix.size(2), offset=1)
        return distance_matrix[:, tri_indices[0], tri_indices[1]]

# ==================== Augmenter 类 ====================
class DataAugmenter:
    """
    文本增强策略管理类，通过名称调用不同增强方法。
    支持：'random_deletion','random_swap','random_insertion','augment_eda'.
    """
    def __init__(self):
        self.methods = {
            'random_deletion': self._random_deletion,
            'random_swap': self._random_swap,
            'random_insertion': self._random_insertion,
            'augment_eda': self._eda,
        }

    def get(self, name):
        return self.methods.get(name, self._identity)

    @staticmethod
    def _identity(entry):
        return [entry]

    @staticmethod
    def _random_deletion(entry, p: float = 0.1):
        vs, cc, lvl, dept = entry
        words = segcut.cut(cc)
        if len(words) <= 1:
            return [entry]
        new_words = [w for w in words if random.random() > p]
        if not new_words:
            new_words = [random.choice(words)]
        return [(vs, ''.join(new_words), lvl, dept)]

    @staticmethod
    def _random_swap(entry, n_swaps: int = 1):
        vs, cc, lvl, dept = entry
        words = segcut.cut(cc)
        length = len(words)
        if length < 2:
            return [entry]
        for _ in range(n_swaps):
            i, j = random.sample(range(length), 2)
            words[i], words[j] = words[j], words[i]
        return [(vs, ''.join(words), lvl, dept)]

    @staticmethod
    def _random_insertion(entry, n_insert: int = 1):
        vs, cc, lvl, dept = entry
        words = segcut.cut(cc)
        if not words:
            return [entry]
        for _ in range(n_insert):
            iw = random.choice(words)
            pos = random.randint(0, len(words))
            words.insert(pos, iw)
        return [(vs, ''.join(words), lvl, dept)]

    def _eda(self, entry):
        funcs = [self._random_deletion, self._random_swap, self._random_insertion]
        func = random.choice(funcs)
        if func is self._random_deletion:
            return func(entry, p=0.1)
        elif func is self._random_swap:
            return func(entry, n_swaps=1)
        else:
            return func(entry, n_insert=1)

# ==================== 长尾清洗、增强、标签修正 ====================

def select_head_ccs_by_coverage(dataset, coverage: float = 0.8):
    cc2count = defaultdict(int)
    for _, cc, _, _ in dataset:
        cc2count[cc] += 1
    sorted_ccs = sorted(cc2count.items(), key=lambda x: x[1], reverse=True)
    total = sum(cc2count.values())
    acc = 0
    head_ccs = set()
    for cc, cnt in sorted_ccs:
        if acc / total >= coverage:
            break
        head_ccs.add(cc)
        acc += cnt
    return head_ccs


def clean_and_correct_labels(
    dataset,
    cc2emb: dict,
    head_ccs: set,
    min_count: int,
    sim_thresh: float = 0.8,
):
    """
    对长尾样本进行相似度检测，视为噪声标签并修正：
    - 若 tail_cc 与某 head_cc 相似度 ≥ sim_thresh 且该 head_cc 出现次数 ≥ min_count
      则将尾部样本标签修正为 head_cc 的多数标签
    否则保留原标签。
    :param min_count: 用于多数投票的头部频次阈值
    """
    # 统计 head 多数标签
    cc2labels = defaultdict(list)
    for vs, cc, lvl, dept in dataset:
        if cc in head_ccs:
            cc2labels[cc].append(dept)
    cc2major = {cc: Counter(labs).most_common(1)[0][0] for cc, labs in cc2labels.items()}

    head_embs = torch.stack([cc2emb[cc] for cc in head_ccs]) if head_ccs else torch.empty(0)
    # emb_head_map = {cc: cc2emb[cc] for cc in head_ccs}

    corrected = []
    for vs, cc, lvl, dept in dataset:
        if cc in head_ccs:
            # 头部不修改
            corrected.append((vs, cc, lvl, dept))
        else:
            emb = cc2emb[cc].unsqueeze(0)
            if head_embs.numel() > 0:
                sims = torch.mm(emb, head_embs.t()).squeeze(0)
                idx = torch.argmax(sims).item()
                sim_val = sims[idx].item()
                head_cc = list(head_ccs)[idx]
                # 如果相似度高且 head_cc 出现次数足够，修正标签
                if sim_val >= sim_thresh and len(cc2labels[head_cc]) >= min_count:
                    new_dept = cc2major[head_cc]
                    corrected.append((vs, cc, lvl, new_dept))
                else:
                    corrected.append((vs, cc, lvl, dept))
            else:
                corrected.append((vs, cc, lvl, dept))
    return corrected

# ==================== 数据集类 ====================
class ChiefCompDataset(Dataset):
    def __init__(self, args, dataset, dataset_name):
        """
        支持头部保留；长尾高相似样本增强；高相似视为噪声标签并修正。
        :param args.min_count: 头部频次最小阈值
        :param args.dynamic_coverage: 头部覆盖率
        :param args.cc_sim_thresh: 相似度阈值
        :param args.label_correct: 是否执行标签修正
        :param args.mode: 'train'/'valid'
        """
        # 编码器初始化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enc = args.text_encoder.strip().lower()
        self.raw_data = dataset
        unique_ccs = list({cc for _, cc, _, _ in dataset})

        if self.enc in ['bert', 'roberta', 'bioclinicalbert']:
            if self.enc == 'bert':
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                self.model = BertModel.from_pretrained('bert-base-chinese').to(self.device)
            elif self.enc == 'roberta':
                self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
                self.model = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(self.device)
            elif self.enc == 'bioclinicalbert':
                self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                self.model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(self.device)                
            self.model.eval()

            # 全量编码 unique cc
            with torch.no_grad():
                encoded = self.tokenizer(unique_ccs, padding=True, truncation=True, return_tensors='pt', max_length=20)
                outputs = self.model(input_ids=encoded['input_ids'].to(self.device), attention_mask=encoded['attention_mask'].to(self.device))
                # emb_all = F.normalize(outputs.last_hidden_state[:, 0, :], dim=1)
            
            if args.backbone == 'TextCNN':
                emb_all =  F.normalize(outputs.last_hidden_state, dim=2) # 用于TextCNN shape: (B, L, D)
            else:
                
                emb_all = F.normalize(outputs.last_hidden_state[:, 0, :], dim=1)# shape: (B, D)  

            self.cc2emb = {cc: emb_all[i] for i, cc in enumerate(unique_ccs)}
        
        # 中文 CLIP 文本编码

        elif self.enc == 'cn_clip':
            
            text_tokens = clip.tokenize(unique_ccs).to(self.device)
            self.cc2emb = {cc: text_tokens[i] for i, cc in enumerate(unique_ccs)}
        else:
            raise ValueError(f"Unsupported encoder: {self.enc}")

        # 标签修正
        if args.label_correct and args.mode in ['train', 'valid']:
            # 头尾划分
            head_ccs = select_head_ccs_by_coverage(dataset, coverage=args.dynamic_coverage)            
            dataset = clean_and_correct_labels(
                dataset,
                cc2emb=self.cc2emb,
                head_ccs=head_ccs,
                min_count=args.min_count,
                sim_thresh=args.cc_sim_thresh
            )

        # 缓存
        self.cache_dir = args.cache_dir
        self.cache_file = os.path.join(self.cache_dir, f'cached_{dataset_name}_{self.enc}.pt')
        if os.path.exists(self.cache_file):
            self.data = torch.load(self.cache_file)
        else:
            self.data = []
            for vs, cc, lvl, dept in dataset:
                vs_tensor = torch.tensor(vs, dtype=torch.float32, device=self.device)
                emb = self.cc2emb.get(cc)
                emb = emb.to(torch.float32)
                self.data.append({'VS': vs_tensor, 
                                  'Level': lvl, 
                                  'CC_tokens': emb, 
                                  'Dept_digit': dept})
            torch.save(self.data, self.cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
