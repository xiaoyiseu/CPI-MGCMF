import argparse

def get_args():
    parser = argparse.ArgumentParser(description="AI Triage")
    ######################## trandformer settings ########################
    parser.add_argument("--in_dim", default=768)#bert:768; clip：512
    parser.add_argument("--embed_dim", default=128)
    parser.add_argument("--num_heads", default=8)
    parser.add_argument("--hidden_dim", default=64)
    parser.add_argument("--nec", default=3, help = 'num_encoder_layers')
    parser.add_argument("--ndc", default=3, help = 'num_decoder_layers')

    ######################## imputation settings ########################
    parser.add_argument("--ImputMode", default='CGMM', 
                        help="'RF', 'GAN', 'MICE', 'VAE', 'CGMM")    
    parser.add_argument("--n_comps", default=2, help="CGMM")

    parser.add_argument("--ImputMode2", default='CGMM',  
                        help="'RF', 'GAN', 'MICE', 'VAE', 'CGMM") 
 
    ######################## general settings ########################
    parser.add_argument("--backbone", default='TextResNet', 
                        help="'Transformer', 'ResNet','TextCNN', 'MLP', 'TextResNet'") 
    parser.add_argument("--grade", default=False, help="Graded training") #False   True
    parser.add_argument("--SFD", default=False, help="Manhattan Distance Feature") #False   True

    parser.add_argument("--text_encoder", default='bert', 
                        help="'bert', 'roberta', 'bioclinicalbert',cn_clip //tokenizer, 'sbert', 'word2vec', 'umls', 'tfidf'")     
    # parser.add_argument("--pca_dd", default=True, help="PCA降维CC")  
    parser.add_argument("--vsEmbed", default=True)  #False   True
    parser.add_argument("--FusionEarly", default=True)  #False   True
    parser.add_argument("--CMF", default=True, help="Cross-Modal Fusion")#False   True
    parser.add_argument("--n_comp", type=int, default=8, help="PCA components")  


    parser.add_argument('-se', type=int, default=2, help='Taylor expansion series') 
    parser.add_argument("--loss", default='', # pdc+ctl+cmc
                        help="which loss to use ['ime', 'seu', 'cmc']") 
    
    ########################       清洗脏标签样本         ########################
    parser.add_argument("--label_correct", default=False, help="训练集中脏标签处理") #False   True
    parser.add_argument("--min_count", default=20, help="尾部标签与头部标签重叠数量") 
    parser.add_argument("--dynamic_coverage", default=0.98, help="头部数据占比") 
    parser.add_argument("--cc_sim_thresh", default=0.8, help="相似度阈值") 

    ########################        长尾数据重采样        ########################
    parser.add_argument('--data_quality', choices=['dirty','clean'], default='dirty',
                        help='先用脏数据(dirty)训练，再用干净数据(clean)微调')
    
    parser.add_argument("--Resample", default=False, help="是否对长尾数据重采样")  #False   True   
    parser.add_argument("--threshold", default=0.5)     
    parser.add_argument("--alpha_vs", default=0.6)   
    parser.add_argument("--alpha_cc", default=1)     
    parser.add_argument("--filter", default=False, help="剔除与原始样本相似度超过阈值的增强样本") 
    ########################    给VS&CC添加mask    ########################
    parser.add_argument("--use_vs_mask", default=False, help="是否给VS添加mask") #False   True    
    parser.add_argument("--vs_mask_rate", default=0.1)   
    parser.add_argument("--text_length", default=20)  
    parser.add_argument("--rand_seed", default=42)  

    ######################## file path settings ########################
    parser.add_argument("--data_path_dirty", default=r"./data/TriageData(raw).txt")
    parser.add_argument("--data_path_clean", default=r"./data/CleanData.txt")
    parser.add_argument("--stopword_path", default=r"./data/stopword.txt")
    parser.add_argument("--cache_dir", default=r'./cached_data/')                                                                                                 
    parser.add_argument("--bestmodel", default=r'./weight/best_model.pth')
    parser.add_argument("--log_path", default=r'./result/train/training_log.txt')
    parser.add_argument("--output_dir", default=r'./result/')
    parser.add_argument("--cache_clip", default=r'./weight/clip_cn_vit-b-16.pt')

    ######################## model general settings ####################
    parser.add_argument("--length", type=int, default=1000000) 
    parser.add_argument("--batch_size", type=int, default=1024)  
    parser.add_argument("--bs_test", type=int, default=512)  
    parser.add_argument("--lstm_layers", type=int, default=1)  

    parser.add_argument("--num_epochs", type=int, default=100)  
    parser.add_argument("--stage0_epochs", type=int, default=6)  
    parser.add_argument("--stage1_epochs", type=int, default=6) 
    parser.add_argument("--lr", type=float, default=1e-3) 
    parser.add_argument("--lr_vs", type=float, default=1e-3) 
    parser.add_argument("--lr_cc", type=float, default=1e-4) 
    parser.add_argument("--patience", type=int, default=10)  
    parser.add_argument("--task", default='Norm', help="'VS2LV', 'CC2DP', 'CC2LV', 'VS2DP', Norm")  
    parser.add_argument("--mode", default='train')  
    parser.add_argument("--CI", default=False)  #False   True
    parser.add_argument("--CIPrior", default=False, help="True:基于先验+条件插值， False:基于先验+伪标签插值") 

    args = parser.parse_args(args = [])   
    return args  

