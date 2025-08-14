import argparse
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)
from scipy.stats import ttest_1samp, sem, t
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

from module.manager import Data_Indices
from utils.options import get_args  # 假设这里注册了 --ml_model

from functools import partial
# from model.SingleMod import TextCNN, RNNClassifier, TransformerClassifier
from model.SingleMod import MLPClassifier, SimpleRNNClassifier, TextCNNClassifier, TransformerClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gc

# 1. 全局过滤掉指定类别的警告
warnings.filterwarnings("ignore", category=FutureWarning)            # HF 文件下载的 FutureWarning
warnings.filterwarnings("ignore", category=UserWarning)              # HF 模型加载的 UserWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)       # sklearn 的 ConvergenceWarning


import os
import joblib

def save_model(m, model_name, task, out_dir="./weight/SingleTask"):
    """
    将模型保存到 out_dir/<task>_<model_name>.pt (深度学习)
                   或 out_dir/<task>_<model_name>.pkl (sklearn)
    """
    out_dir = os.path.join(out_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{task}_{model_name}"
    
    if isinstance(m, torch.nn.Module):
        # 保存 state_dict
        path = os.path.join(out_dir, filename + ".pt")
        torch.save(m.state_dict(), path)
#         print(f"Saved PyTorch model to {path}")
    else:
        # sklearn 模型
        path = os.path.join(out_dir, filename + ".pkl")
        joblib.dump(m, path)
#         print(f"Saved sklearn model to {path}")

def loader_to_numpy(loader, task='severity'):
    X_list, y_list = [], []
    for batch in loader:
        if task == 'severity':
            vs = batch['VS'].cpu().numpy()
            lvl = batch['Level'].cpu().numpy()
            X_list.append(vs.reshape(vs.shape[0], -1))
            y_list.append(lvl)
        elif task == 'department':
            cc = batch['CC_tokens'].cpu().numpy()
            dpt = batch['Dept_digit'].cpu().numpy()
            X_list.append(cc.reshape(cc.shape[0], -1))
            y_list.append(dpt)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def get_model_factory(num_class, feat_dim=None, hidden_dim=256):
    return {
        # VS 
        'XGBoost': lambda: XGBClassifier(
            objective='multi:softprob', num_class=num_class,
            eval_metric='mlogloss', n_estimators=300,
            max_depth=6, learning_rate=1e-3, random_state=42),
        'LightGBM': lambda: LGBMClassifier(
            objective='multiclass', num_class=num_class,
            n_estimators=300, max_depth=6,
            min_data_in_leaf=5, min_gain_to_split=0.0,
            force_col_wise=True, learning_rate=1e-3, random_state=42),
        'SVM': lambda: SVC(kernel='rbf', probability=True, random_state=42),
        'RF': lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        'Logistic': lambda: LogisticRegression(
            multi_class='multinomial', solver='lbfgs',
            max_iter=200, random_state=42),
        'CatBoost': lambda: CatBoostClassifier(
            iterations=300, depth=6, learning_rate=1e-3,
            loss_function='MultiClass', verbose=False, random_seed=42),
        'GNB': lambda: GaussianNB(),
        'KNN': lambda: neighbors.KNeighborsClassifier(n_neighbors=5),

        # CC 
        'MLP': lambda: MLPClassifier(feat_dim, hidden_dim, num_class),
        'RNN': lambda: SimpleRNNClassifier(feat_dim, hidden_dim, num_class, rnn_type='rnn'),
        'LSTM':lambda: SimpleRNNClassifier(feat_dim, hidden_dim, num_class, rnn_type='lstm'),
        'GRU': lambda: SimpleRNNClassifier(feat_dim, hidden_dim, num_class, rnn_type='gru'),
        'TextCNN': lambda: TextCNNClassifier(input_dim=feat_dim, num_filters=128, kernel_sizes=[3,5,7], num_class=num_class, dropout=0.1),
        'Transformer': lambda: TransformerClassifier(input_dim=feat_dim, nhead=4, nlayers=2, num_class=num_class, dim_feedforward=512, dropout=0.1),
    }

def train_deep_model(model, X_tr, y_tr, X_val, y_val,
                     num_epochs=100, lr=1e-3, batch_size=64,
                     patience=3, device=None):
    """
    深度模型训练：支持训练/验证拆分、EarlyStopping、LR调度，并在结束后释放显存
    返回对验证集的预测 y_pred
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)

    # 构造 DataLoader
    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    val_ds   = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    for epoch in range(1, num_epochs+1):
        # training
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                total_val_loss += loss.item() * xb.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        # scheduler and early stop
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_weights = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # restore best model
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def repeated_experiments(X, y, build_model, model_name=None, task_name='severity',
                         repeats=5, test_size=0.2, num_epochs=100, lr=1e-3, batch_size=64,
                         patience=5, device=None):
    """
    重复实验，对深度和传统模型分别调用训练逻辑，实验后清理内存
    返回各项指标 numpy arrays
    """
    import numpy as _np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, cohen_kappa_score, confusion_matrix
    )
    from scipy.stats import ttest_1samp

    metrics = {k: [] for k in
               ['accuracy','precision_w','precision_m','f1_w','f1_m',
                'sensitivity','specificity','kappa']}
    deep_models = ['MLPClassifier', 'TextCNNClassifier', 'SimpleRNNClassifier', 'TransformerClassifier']
    for i in range(repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42+i, stratify=y)
        
        if task_name=='department':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = build_model().to(device)
            name = type(m).__name__
    
            if name in deep_models:
                # deep branch remains same
                split = int(0.8 * len(X_tr))
                X_tr_d, X_val_d = X_tr[:split], X_tr[split:]
                y_tr_d, y_val_d = y_tr[:split], y_tr[split:]
                m = train_deep_model(
                    m, X_tr_d, y_tr_d, X_val_d, y_val_d,
                    num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                    patience=patience, device=device)
                # inference
                m.eval()
                with torch.no_grad():
                    X_te_tensor = torch.FloatTensor(X_te).to(device)
                    preds = m(X_te_tensor)
                    y_pred = preds.argmax(dim=1).cpu().numpy()
                save_model(m, model_name, task_name)
        else:
            m = build_model()
            # traditional ML
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            save_model(m, model_name, task_name)
            
        # cleanup model and GPU
        try:
            del model
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

        # compute metrics
        acc = accuracy_score(y_te, y_pred)
        prec_m = precision_score(y_te, y_pred, average='macro', zero_division=0)
        rec    = recall_score(y_te, y_pred, average='macro', zero_division=0)
        f1_m   = f1_score(y_te, y_pred, average='macro', zero_division=0)
        prec_w = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        f1_w   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
        kappa  = cohen_kappa_score(y_te, y_pred)

        cm = confusion_matrix(y_te, y_pred)
        specs = []
        for c in range(cm.shape[0]):
            TP = cm[c,c]
            FP = cm[:,c].sum() - TP
            TN = cm.sum() - TP - FP - (cm[c,:].sum() - TP)
            specs.append(TN/(TN+FP) if TN+FP>0 else 0)
        spec = _np.mean(specs)

        metrics['accuracy'].append(acc)
        metrics['precision_m'].append(prec_m)
        metrics['f1_m'].append(f1_m)
        metrics['precision_w'].append(prec_w)
        metrics['f1_w'].append(f1_w)
        metrics['sensitivity'].append(rec)
        metrics['specificity'].append(spec)
        metrics['kappa'].append(kappa)

    # 转 numpy 返回
    return {k: _np.array(v) for k,v in metrics.items()}




def main(model_names, train_loader, valid_loader, task='severity', 
         repeats=5, test_size=0.2, 
         num_epochs=100, lr=1e-3, 
         batch_size=64, patience=5):
    
    # 准备好表头，只打印一次
    col_widths = {
        'method':10, 'accuracy':20, 'sensitivity':12, 'specificity':12, 
        'precision':15,'f1-score':15, 'kappa':8, 'p-value':10
    }
    headers = list(col_widths.keys())
    sep = '+' + '+'.join('-'*col_widths[h] for h in headers) + '+'
    header_row = '|' + '|'.join(f"{h:^{col_widths[h]}}" for h in headers) + '|'
    print(sep)
    print(header_row)
    print(sep)
    
    # 依次跑不同模型
    for model_name in model_names:

        X_tr, y_tr = loader_to_numpy(train_loader, task=task)
        X_val, y_val = loader_to_numpy(valid_loader, task=task)
        X_all = np.vstack([X_tr, X_val])
        y_all = np.hstack([y_tr, y_val])
        num_severity = int(np.unique(y_all).size)
        
        # 文本任务需要 vocab_size
        if task == 'department':
            sample = next(iter(train_loader))
            feat_dim = sample['CC_tokens'].shape[1]  # 或 sample['VS'].shape[1]
        else:
            feat_dim = None

        factories = get_model_factory(num_class=num_severity, feat_dim=feat_dim)
        results = repeated_experiments(X_all, y_all, 
                                       factories[model_name], model_name = model_name, task_name=task,
                                       repeats=repeats, test_size=test_size, num_epochs=num_epochs, 
                                       lr=lr, batch_size=batch_size,patience=patience)

        # 统计
        mean_acc = results['accuracy'].mean()
        se_acc = sem(results['accuracy'])
        df = len(results['accuracy'])-1
        t_crit = t.ppf(0.975, df)
        ci_low = mean_acc - t_crit*se_acc
        ci_up  = mean_acc + t_crit*se_acc
        baseline = 1.0/num_severity
        t_stat, p_val = ttest_1samp(results['accuracy'], baseline)

        row = [
            model_name,
            f"{mean_acc*100:.2f}[{ci_low*100:.2f},{ci_up*100:.2f}]",
            f"{results['sensitivity'].mean()*100:.2f}",
            f"{results['specificity'].mean()*100:.2f}",            
            f"{results['precision_w'].mean()*100:.2f}/{results['precision_m'].mean()*100:.2f}",
            f"{results['f1_w'].mean()*100:.2f}/{results['f1_m'].mean()*100:.2f}",

            f"{results['kappa'].mean():.3f}",
            f"{p_val:.2e}"]
        data_row = '|' + '|'.join(f"{v:^{col_widths[h]}}" for v,h in zip(row, headers)) + '|'
        print(data_row)
    # 底部分隔
    print(sep)
    
if __name__ == "__main__":
    args = get_args()
    print('编码方式：', args.text_encoder)
    for task in ['severity', 'department']:
#         task='department'#severity, department
        if task=='severity':
            model_name=['XGBoost','RF','Logistic','SVM','CatBoost', 'KNN']#'lightgbm','GNB'
        elif task=='department':
            model_name=['MLP','RNN','LSTM','GRU','TextCNN', 'Transformer']
    
        # 提取并合并样本
        train_loader, valid_loader, _, _ = Data_Indices(args)
        main(model_name, train_loader, valid_loader, task=task,
             repeats=5, test_size=0.2, 
             num_epochs=100, lr=1e-3, 
             batch_size=512, patience=5)