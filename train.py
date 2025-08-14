import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import logging
from utils.options import get_args
import os, time
from model.CrossAttention import Transformer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from module.TrainValid import train, evaluate, print_metrics
from module.manager import Data_Indices
from utils.iotools import save_train_configs
from utils.visualization import plot_pr_curve
import warnings
warnings.filterwarnings('ignore')

def Result(meters, batch_size):
    loss = meters['loss'].avg 
    seu_loss = meters['seu_loss'].avg 
    acc_s = meters['correct_s'].avg / batch_size 
    acc_d = meters['correct_d'].avg / batch_size 
    return loss, seu_loss, acc_s, acc_d

def cls_trans(label):
    no_of_classes = int(label.max().item()) + 1 
    valid_label = label[label >= 0]
    samples_per_cls = torch.bincount(valid_label, minlength=no_of_classes).tolist()
    return no_of_classes, samples_per_cls


def set_requires_grad(model, module_name_list, requires_grad: bool):
    """把 model 里名字包含 module_name_list 中任一 key 的参数，设置 requires_grad。"""
    for name, p in model.named_parameters():
        if any(name.startswith(m) for m in module_name_list):
            p.requires_grad = requires_grad

def get_cc_modules_to_unfreeze(args):
    mods = []
    # 文本特征提取器部分
    if args.backbone == 'Transformer':
        mods += ['encoder', 'decoder']
    elif args.backbone == 'ResNet':
        mods += ['resnet']
    elif args.backbone == 'TextCNN':
        mods += ['textcnn']
    elif args.backbone == 'TextResNet':
        mods += ['textresnet']
    # 最终 Dept 分类头
    mods += ['fc_department']
    # 如果用了 CMF，还要解冻融合路径
    if args.CMF:
        mods += ['resnet_fus', 'cross_att1', 'cross_att2']
    return mods

def build_name(args):
    name_parts = [args.backbone, args.ImputMode]
    optional = []
    if args.vsEmbed:
        optional.append(f'vsEmbed')
    if args.SFD:
        optional.append(f'SFD')
    if args.FusionEarly:
        optional.append(f'FusEarly{args.n_comp}')
    if args.Resample:
        optional.append(f'Resample')
    if args.loss:
        optional.append(f'loss{args.loss}')
    if args.use_vs_mask:
        optional.append('Mask')
    if args.CMF:
        optional.append('CMF')
    if getattr(args, 'text_encoder', None):
        optional.append(str(args.text_encoder))
    if len(optional) == 1:
        name_parts.append(optional[0])
    elif len(optional) > 1:
        name_parts.append('_'.join(optional))
    return '_'.join(name_parts)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data_quality == 'dirty':
        output_result = os.path.join(args.output_dir, 'dirty_phase')
    else:  # clean
        output_result = os.path.join(args.output_dir, 'clean_phase')

    os.makedirs(output_result, exist_ok=True)
    train_loader, valid_loader, test_loader, Y1, Y2, dic1, dic2 = Data_Indices(args)

    num_sevty, per_cls_vs = cls_trans(Y1)
    num_deprt, per_cls_cc = cls_trans(Y2)

    # base_vs = 40 if args.SFD and args.FusionEarly else 29

    if args.task in ['VS2LV', 'CC2DP', 'CC2LV', 'VS2DP']:
        input_dim_vs = 29
        input_dim_cc = args.in_dim
    elif args.task=='Norm':
        base_vs = 29 if args.SFD and args.FusionEarly else 29#64
        base_cc = args.in_dim
        if args.FusionEarly:
            input_dim_vs = base_vs + args.n_comp
            input_dim_cc = base_cc + args.n_comp
        else:
            input_dim_vs = base_vs
            input_dim_cc = base_cc

    model = Transformer(args,
                        input_dim=input_dim_cc, 
                        embed_dim=args.embed_dim, #128
                        num_heads=args.num_heads, #2
                        hidden_dim=args.hidden_dim, #256
                        num_encoder_layers=args.nec, #4
                        num_decoder_layers=args.ndc, #4
                        output_dim_s=num_sevty, #4
                        output_dim_d=num_deprt,
                        per_cls_vs=per_cls_vs,
                        per_cls_cc=per_cls_cc,
                        input_dim_vs=input_dim_vs,
                        ).to(device)

    NAME = build_name(args)
    testresult_path = os.path.join(output_result, NAME)
    output_path = os.path.join(testresult_path, str(args.rand_seed))
    
    bestmodel = os.path.join(output_path, args.bestmodel)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.dirname(bestmodel), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    if args.data_quality == 'clean':
        premodel_dir = output_path.replace('clean_phase','dirty_phase')
        dirty_model_path = os.path.join(premodel_dir, args.bestmodel)
        clean_model_path = os.path.join(output_path, args.bestmodel)
        print(f"Load pretrained weights from {dirty_model_path}")
        checkpoint = torch.load(dirty_model_path)
        state_dict = checkpoint.state_dict() if isinstance(checkpoint, nn.Module) else checkpoint
        vs_sd = {k: v for k, v in state_dict.items() if k.startswith('VitalEmbed')}
        model.load_state_dict(vs_sd, strict=False)

        bestmodel = clean_model_path
        optimizer = optim.AdamW([{'params': model.VitalEmbed.parameters(), 'lr': args.lr_vs},
                {'params': [p for n, p in model.named_parameters() if not n.startswith('VitalEmbed')],
                'lr': args.lr_cc}],weight_decay=1e-5)
    else:
        # optimizer = optim.AdamW([{'params': model.VitalEmbed.parameters(), 'lr': args.lr_vs},
        #         {'params': [p for n, p in model.named_parameters() if not n.startswith('VitalEmbed')],
        #         'lr': args.lr_cc}],weight_decay=5e-5)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0)

    args.log_path = os.path.join(output_path, 'training_log.txt')
    save_train_configs(testresult_path, args)

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)     
    logging.basicConfig(filename=args.log_path, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(message)s', 
                        filemode='a') 
    logger = logging.getLogger()

    best_score = 0.0  
    best_model_info = {}  
    early_stop_counter = 0 
    train_loss2, valid_loss, valid_acc_severity, valid_acc_department = [], [], [], []

    if args.data_quality == 'clean':
        vs_modules = ['VitalEmbed', 'fc_severity']  
        cmf_modules = ['resnet_fus', 'cross_att1', 'cross_att2']
        cc_modules = get_cc_modules_to_unfreeze(args) 

        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if any(name.startswith(m) for m in vs_modules):
                p.requires_grad = True

    current_stage = 0
    for epoch in range(1, args.num_epochs + 1):
        if args.data_quality == 'clean':
            if epoch == args.stage0_epochs + 1 and current_stage == 0:
                current_stage = 1
                # Stage1
                for name, p in model.named_parameters():
                    if any(name.startswith(m) for m in cmf_modules):
                        p.requires_grad = True
                print(f"→ Stage1 @ epoch {epoch}: unfreeze CMF modules {cmf_modules}")
            if epoch == args.stage0_epochs + args.stage1_epochs + 1 and current_stage == 1:
                current_stage = 2
                # Stage2
                for name, p in model.named_parameters():
                    if any(name.startswith(m) for m in cc_modules):
                        p.requires_grad = True
                print(f"→ Stage2 @ epoch {epoch}: unfreeze CC modules {cc_modules}")

        start_time = time.time()  
        args.mode = 'train'
        meters_train = train(args, train_loader, model, optimizer, scheduler)
        args.mode = 'valid'
        meters_val = evaluate(args, valid_loader, model)

        train_loss,seu_loss1, train_acc_s, train_acc_d = Result(meters_train, args.batch_size)
        val_loss, seu_loss2, val_acc_s, val_acc_d = Result(meters_val, args.batch_size)
        combined_score = val_acc_s + 0.1 * val_acc_d

        if val_acc_d > best_score:
            best_score = val_acc_d
            # best_model_state = model.state_dict()
            torch.save(model, bestmodel)

            best_model_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'seu_loss1': seu_loss1,
                'seu_loss2': seu_loss2,
                'val_acc_s': val_acc_s,
                'val_acc_d': val_acc_d,
                'combined_score': combined_score
            }
            early_stop_counter = 0 
        else:
            early_stop_counter += 1 

        if early_stop_counter >= args.patience :
            logger.info(f"Training Finished! Early stopping at epoch {epoch}") 
            break 

        train_loss2.append(train_loss)
        valid_loss.append(val_loss)
        valid_acc_severity.append(val_acc_s)
        valid_acc_department.append(val_acc_d)

        epoch_time = time.time() - start_time
        log_message = (f'Epoch:{epoch:03d}, '
                    f'train loss:{train_loss:.4f}, '
                    f'val loss:{val_loss:.4f}, '
                    f'seu_loss1:{seu_loss1:.4f}, '
                    f'seu_loss2:{seu_loss2:.4f}, '
                    f'val_sev_acc:{val_acc_s*100:.2f}, '
                    f'val_dep_acc:{val_acc_d*100:.2f}, '
                    f'time:{epoch_time:.2f} s')
        logger.info(log_message) 
        print(log_message) 

    logger.info(f"Best model found at epoch {best_model_info['epoch']}. \n"
                f"Train Loss: {best_model_info['train_loss']:.4f}, "
                f"Val Loss: {best_model_info['val_loss']:.4f}, "
                f"Val Severity Acc: {best_model_info['val_acc_s']:.4f}, "
                f"Val Depart Acc: {best_model_info['val_acc_d']:.4f}, "
                f"Combined Score: {best_model_info['combined_score']:.4f}")
    

    ## 测试
    model = torch.load(bestmodel)
    args.mode = 'test'
    dic_path = os.path.join(args.cache_dir, 'inverse_dicts.pth')
    ckpt = torch.load(dic_path)
    dic1, dic2 = ckpt['inverse_dic1'],ckpt['inverse_dic2']

    # test_loader =


    _, metrics, avg_time, true_s, probs_s, true_d, probs_d, cc_feat, vs_feat = evaluate(args, test_loader, model)
    
    print_metrics(metrics, 
                file_path=os.path.join(testresult_path,'test.txt'), 
                mode = args.mode)
    
    print(f"Ave time: {avg_time:.4f} ms")

    plot_pr_curve(
        true_labels=true_s,
        pred_probs=probs_s,
        class_dict = dic1,
        task_name="sevty",
        file_path=os.path.join(output_path, 'pr_curve_sevty.svg')
    )
    plot_pr_curve(
        true_labels=true_d,
        pred_probs=probs_d,
        class_dict = dic2,
        task_name="depat",
        file_path=os.path.join(output_path, 'pr_curve_depat.svg')
    )
    return os.path.join(testresult_path,'test.txt')

def ConfidenceInterval(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    metrics_data = {'Sevty': [], 'Depat': []}
    for line in lines:
        parts = line.strip().split()
        if not parts or parts[0] not in metrics_data:
            continue
        metric = parts[0]
        metrics_data[metric].append({
            'accuracy': float(parts[1]),
            'f1_m': float(parts[2]),
            'f1_w': float(parts[3]),
            'kappa': float(parts[4]),
            'prec_m': float(parts[5]),
            'prec_w': float(parts[6]),
            'sens': float(parts[7]),
            'spec': float(parts[8])
        })

    # 计算均值与 95% 置信区间
    t_value = 2.776  # df=4, 95% 置信水平
    results = []
    for metric, records in metrics_data.items():
        df = pd.DataFrame(records)
        n = len(df)
        means = df.mean()
        stds = df.std(ddof=1)
        ci_low = means - t_value * stds / np.sqrt(n)
        ci_high = means + t_value * stds / np.sqrt(n)

        results.append({
            'metric': metric,
            'accuracy': f"{means['accuracy']*100:.2f}[{ci_low['accuracy']*100:.2f},{ci_high['accuracy']*100:.2f}]",
            'sensitivity': f"{means['sens']*100:.2f}",
            'specificity': f"{means['spec']*100:.2f}",
            'precision': f"{means['prec_w']*100:.2f}/{means['prec_m']*100:.2f}",
            'f1-score': f"{means['f1_w']*100:.2f}/{means['f1_m']*100:.2f}",
            'kappa': f"{means['kappa']:.3f}",
    #         'p-value': '<compute separately>'
        })

    result_df = pd.DataFrame(results).set_index('metric')
    return result_df

if __name__ == '__main__':
    args = get_args()
    train_mode = '1k-fold' #'5k-fold'
    if train_mode == '5k-fold':
        repeats = 5
        # for n_comp in range(2, 12, 2):
        #     args.n_comp = n_comp
        for i in range(repeats):
            # print(f'经PCA降维后的维度： {n_comp}')
            print(f'===========> Round {i+1}')
            args.rand_seed=42+i
            file_path = main(args)
        result_df = ConfidenceInterval(file_path)
        print(result_df)

        dirpath = os.path.dirname(file_path)
        new_filename = 'results.txt'
        new_path = os.path.join(dirpath, new_filename)
        result_df.to_csv(new_path, sep="\t", index=True)

    else:
        file_path = main(args)


