# Conditional Probability Imputation and Cross-Modal Fusion under Incomplete and Heterogeneous Data
Accurate and timely triage in emergency departments (ED) requires decision support that is robust to incomplete and heterogeneous data. This study proposes a knowledge-guided triage framework that addresses two critical challenges:   

(i) handling missing structured features (e.g., vital signs, age, and mode of arrival) through an uncertainty-aware Conditional-Probability Imputation (CPI) scheme. 

(ii) integrating imputed structured features with concise clinician-recorded chief complaints via a multigranularity cross-modal fusion pipeline.

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/d3eda3fd-6fd7-4a8c-b2bf-275e464dbf3d" />

### 1.Raw Data and Tasks  
1.1 Triage Data     
chief complants, vital signs (e.g., temperature, pulse rate, respiratory rate, blood pressure, oxygen saturation), demographic details (e.g., sex, age), and mode of arrival.

1.2 Tasks    
Task 1-Severity Level Prediction：Level 4 (least critical) to Level 1 (most critical)       
Task 2-Department Recommendation: Surgery, Internal Medicine, Neurology, Otolaryngology (ENT), Obstetrics, Ophthalmology, Gynecology, Orthopedics, Trauma Center, and Neurosurgery.   

Per-class sample counts and percentages for severity levels and for the departments      
<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/f206aff3-758d-4567-ac33-79fd5f7b0406" />

### 2.Correlation analysis
Correlation analysis between structured data variables and labels (Severity and Department) using the Mantel test

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/4e4f879a-20c7-4040-a9ca-c35609cb1565" />

### 3.Requirements
All experiments were conducted on a NVIDIA A6000 GPU for training and evaluation. The initial learning rate was set to 0.001, and AdamW was used as the optimizer. A CosineAnnealingLR scheduler was employed to adjust the learning rate during training. 
```
python 3.8.19
torch 2.3.0+cu121
torchvision 0.18.0
numpy 1.22.1
pandas 2.0.3
pillow 10.3.0
pkuseg 0.0.25
BERT ‘bert-base-chinese’ (HuggingFace)
```

### 4.Training & Testing
4.1 Experiments (Ablation & Comparison)
```
python train.py
python test.py
```
4.2 Machine learning models   

Train & Test
```
python SingleTask(ML).py
```

### 5.Metrics & Results
5.1 Metrics     
Accuracy (95% CI), Sensitivity (SENS), Specificity (SPEC), Precision (PREC), F1-score (F1), and Cohen’s Kappa coefficient (K)

5.2 Ablation experiments      
The folder 'AblayExp' contains training/validation/testing logs.
<img width="800" height="350" alt="image" src="https://github.com/user-attachments/assets/9b2becbc-e725-47be-9203-bb40f3a279fb" />







