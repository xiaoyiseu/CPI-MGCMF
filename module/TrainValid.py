import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.comm import synchronize
from utils.meter import AverageMeter
import time
from sklearn.metrics import *
from scipy.stats import ttest_ind
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torch
from torch.utils.data.dataloader import default_collate
from module.manager import collate_fn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args, train_loader, model, optimizer, scheduler):
    if args.grade:
        head_threshold, head_subset, tail_subset = preprocess_loader(train_loader.dataset, quantile = args.quantile)
        print(f"Head threshold: {head_threshold}, Head samples: {len(head_subset)}, Tail samples: {len(tail_subset)}")
        batch_size = train_loader.batch_size
        num_workers = train_loader.num_workers if hasattr(train_loader, 'num_workers') else 4  # 默认使用4个工作线程
        head_loader = create_dataloader(head_subset, batch_size=batch_size, shuffle=True, 
                                        num_workers=num_workers, collate_fn=collate_fn)
        tail_loader = create_dataloader(tail_subset, batch_size=batch_size, shuffle=True, 
                                        num_workers=num_workers, collate_fn=collate_fn)
        meters_train = train_single_loader(args, head_loader, model, optimizer, scheduler)
        meters_train = train_single_loader(args, tail_loader, model, optimizer, scheduler)
    else:
        meters_train = train_single_loader(args, train_loader, model, optimizer, scheduler)
    return meters_train

def preprocess_loader(dataset, quantile=0.3):
    label_count = defaultdict(int)
    for sample in dataset:
        if isinstance(sample, dict) and "Dept_digit" in sample:
            label = sample["Dept_digit"]
        else:
            raise ValueError(f"Dataset format not supported. Found sample: {sample}")

        label_count[label.item()] += 1 
    counts = np.array(list(label_count.values()))
    threshold = np.percentile(counts, quantile * 100)
    head_indices, tail_indices = [], []
    for idx, sample in enumerate(dataset):
        if isinstance(sample, dict) and "Dept_digit" in sample:
            label = sample["Dept_digit"]
        else:
            raise ValueError(f"Dataset format not supported. Found sample: {sample}")
        if label_count[label.item()] > threshold:
            head_indices.append(idx)
        else:
            tail_indices.append(idx)

    return threshold, Subset(dataset, head_indices), Subset(dataset, tail_indices)

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, collate_fn=collate_fn)

def train_single_loader(args, train_loader, model, optimizer, scheduler):
    model.train()
    meters_train = {
        "loss": AverageMeter(),
        "kl_loss": AverageMeter(),
        "cmc_loss": AverageMeter(),
        "cel_loss": AverageMeter(),
        "seu_loss": AverageMeter(),
        "cont_loss": AverageMeter(),
        "correct_s": AverageMeter(),
        "correct_d": AverageMeter(),
    }
    for n_iter, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()
        optimizer.zero_grad()
        batch["n_iter"] = n_iter
        batch["mode"] = 'train'
        
        ret, _, _ = model(batch)
        total_loss = sum([v for k, v in ret.items() if "loss" in k])

        batch_size = batch['CC_tokens'].shape[0]
        meters_train['loss'].update(total_loss.item(), batch_size)
        meters_train['kl_loss'].update(ret.get('kl_loss', 0), batch_size)
        meters_train['cmc_loss'].update(ret.get('cmc_loss', 0), batch_size)
        meters_train['cel_loss'].update(ret.get('cel_loss', 0), batch_size)
        meters_train['seu_loss'].update(ret.get('seu_loss', 0), batch_size)
        meters_train['cont_loss'].update(ret.get('cont_loss', 0), batch_size)
        meters_train['correct_s'].update(ret.get('correct_s', 0), batch_size)
        meters_train['correct_d'].update(ret.get('correct_d', 0), batch_size)

        total_loss.backward()
        optimizer.step()
        synchronize()
    scheduler.step()
    return meters_train


def evaluate(args, valid_loader, model):
    model.eval()
    
    meters_val = {
        "loss": AverageMeter(),
        "kl_loss": AverageMeter(),
        "cmc_loss": AverageMeter(),
        "cel_loss": AverageMeter(),
        "seu_loss": AverageMeter(),
        "cont_loss": AverageMeter(),
        "correct_s": AverageMeter(),
        "correct_d": AverageMeter()
    }
    
    classification_metrics = {}
    all_probs_s, all_probs_d = [], []  # 存储概率分布
    all_labels_s, all_preds_s = [], []
    all_labels_d, all_preds_d = [], []
    processing_times = []  # 用于记录每条数据的处理时间
    
    cc_feat = []
    vs_feat = []
    with torch.no_grad():
        for n_iter, batch in enumerate(valid_loader):
            start_time = time.time()  
            
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["n_iter"] = n_iter
            batch["mode"] = args.mode
            ret, fusion1_cc, vs0 = model(batch)
            cc_feat.append(fusion1_cc)
            vs_feat.append(vs0)

            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            batch_size = batch['CC_tokens'].shape[0]
            
            meters_val['loss'].update(total_loss.item(), batch_size)
            meters_val['kl_loss'].update(ret.get('kl_loss', 0), batch_size)
            meters_val['cmc_loss'].update(ret.get('cmc_loss', 0), batch_size)
            meters_val['cel_loss'].update(ret.get('cel_loss', 0), batch_size)
            meters_val['seu_loss'].update(ret.get('seu_loss', 0), batch_size)
            meters_val['cont_loss'].update(ret.get('cont_loss', 0), batch_size)
            
            meters_val['correct_s'].update(ret.get('correct_s', 0), batch_size)
            meters_val['correct_d'].update(ret.get('correct_d', 0), batch_size)
            
            all_labels_s.extend(batch['Level'].cpu().numpy())
            all_preds_s.extend(ret.get('sevty_out').argmax(dim=1).cpu().numpy())
            all_labels_d.extend(batch['Dept_digit'].cpu().numpy())
            all_preds_d.extend(ret.get('depat_out').argmax(dim=1).cpu().numpy())
            
            elapsed_time = time.time() - start_time
            processing_times.append(elapsed_time / batch_size)

            if args.mode == 'test':
                all_probs_s.extend(torch.softmax(ret.get('sevty_out'), dim=1).cpu().numpy())  # 保存概率
                all_probs_d.extend(torch.softmax(ret.get('depat_out'), dim=1).cpu().numpy())  # 保存概率

    classification_metrics.update(
        calc_metrics(
            all_labels_s, 
            all_preds_s, 
            "sevty"))
    classification_metrics.update(
        calc_metrics(
            all_labels_d, 
            all_preds_d,
            "depat"))
    
    total_time = sum(processing_times)
    avg_time_per_sample = total_time * 1000 / len(processing_times)
    if args.mode == 'train' or args.mode == 'valid':
        return meters_val
    else:
        return meters_val, classification_metrics, avg_time_per_sample, all_labels_s, all_probs_s, all_labels_d, all_probs_d, cc_feat, vs_feat

def specificity(y_te, y_pred):
    cm = confusion_matrix(y_te, y_pred)
    specs = []
    for c in range(cm.shape[0]):
        TP = cm[c,c]
        FP = cm[:,c].sum() - TP
        TN = cm.sum() - TP - FP - (cm[c,:].sum() - TP)
        specs.append(TN/(TN+FP) if TN+FP>0 else 0)
    spec = np.mean(specs)
    return spec

def calc_metrics(true_labels, pred_labels, task_name):
    return {
        f"{task_name}_acc": accuracy_score(true_labels, pred_labels),
        f"{task_name}_prec_m": precision_score(true_labels, pred_labels, average='macro', zero_division=0),
        f"{task_name}_prec_w": precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
        f"{task_name}_f1_m": f1_score(true_labels, pred_labels, average='macro', zero_division=0),
        f"{task_name}_f1_w": f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
        f"{task_name}_sens": recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
        f"{task_name}_spec": specificity(true_labels, pred_labels),
        f"{task_name}_kappa": cohen_kappa_score(true_labels, pred_labels)
    }

def print_metrics(metrics, file_path=None, mode=None):
    lines = []
    if mode == "test":
        all_metrics = sorted({key.split("_", 1)[1] for key in metrics})
        header = "\t".join(["Metric"] + all_metrics)
        lines.append(header)
        tasks = ["sevty", "depat"]
        for task in tasks:
            task_values = [f"{metrics.get(f'{task}_{metric}', 0):.4f}" if isinstance(metrics.get(f'{task}_{metric}', 0), 
                                                                                     float) else str(metrics.get(f'{task}_{metric}', 0))
                           for metric in all_metrics]
            lines.append("\t".join([task.capitalize()] + task_values))
    else:
        lines.append("Evaluation Metrics:")
        lines.append("*" * 40)
        tasks = {}
        for key, value in metrics.items():
            task, metric = key.split("_", 1)
            tasks.setdefault(task, {})[metric] = value

        for task, task_metrics in tasks.items():
            lines.append(f"Task: {task.capitalize()}")
            lines.append(f"{'Metric':<30}{'Value':<15}")
            lines.append("-" * 40)
            for metric, value in task_metrics.items():
                if isinstance(value, float):
                    lines.append(f"{metric:<30}{value:<15.4f}")
                else:
                    lines.append(f"{metric:<30}{value}")
            lines.append("-" * 40)
    output = "\n".join(lines)
    print(output)
    state = "a" #"w" if mode == "test" else "a"
    if file_path:
        with open(file_path, state, encoding="utf-8") as f:
            f.write(output + "\n")
