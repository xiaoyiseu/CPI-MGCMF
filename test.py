import torch, os, time
from module.TrainValid import evaluate, print_metrics
from utils.options import get_args
from model.CrossAttention import FeatureExtractor
from module.manager import Data_Indices
from utils.visualization import plot_pr_curve
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

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


if __name__ == '__main__':
    args = get_args()
    NAME = build_name(args)
    output_result = os.path.join(args.output_dir, 'dirty_phase')
    testresult_path = os.path.join(output_result, NAME)
    output_path = os.path.join(testresult_path, str(args.rand_seed))

    bestmodel = os.path.join(output_path, args.bestmodel)
    model = torch.load(bestmodel)

    args.mode = 'test'
    start_time = time.time()  
    train_loader, valid_loader, test_loader, Y1, Y2, dic1, dic2 = Data_Indices(args)
    # test_loader, dic1, dic2 = Data_Indices(args)
    end_time = time.time() 

    meters_test, classification_metrics, RecogTine, true_s, probs_s, true_d, probs_d, cc_feat, vs_feat = evaluate(args, test_loader, model)
    print_metrics(classification_metrics, 
                  file_path=os.path.join(args.output_dir,'test.txt'), mode = args.mode)

    N = len(true_s)
    EncTime = (end_time - start_time)/N
    print(f"测试集数量:  {N}")    
    print(f"Encode time: {EncTime:.4f} s, Recog time: {RecogTine:.4f} ms, total time: {EncTime*1000 + RecogTine:.4f} ms")


    plot_pr_curve(
        true_labels=true_s,
        pred_probs=probs_s,
        class_dict = dic1,
        task_name="Severity",
        file_path=os.path.join(args.output_dir, 'pr_curve_severity.svg')
    )
    plot_pr_curve(
        true_labels=true_d,
        pred_probs=probs_d,
        class_dict = dic2,
        task_name="Depart",
        file_path=os.path.join(args.output_dir, 'pr_curve_department.svg')
    )



DepartLib={'Internal Medicine':'IM', 
               'Obstetrics':'OB', 
               'Surgery':'SURG', 
               'Ophthalmology':'OPHTH', 
               'Gynecology':'GYN', 
               'Otolaryngology':'ORL',  
               'Neurosurgery':'NS', 
               'Trauma Center':'TC',
               'Orthopedics':'OPS',
               'Neurology':'NL'
              }   