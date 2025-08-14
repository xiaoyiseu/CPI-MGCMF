import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import argparse
from sklearn.preprocessing import MinMaxScaler
import os, pickle
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_bp(bp):
    if pd.isna(bp):
        return np.nan, np.nan
    try:
        if '/' in bp:
            parts = bp.split('/')
            systolic = int(parts[0]) if parts[0] else np.nan
            diastolic = int(parts[1]) if parts[1] else np.nan
            return systolic, diastolic
    except ValueError:
        return np.nan, np.nan

def preprocess_data(data):
    data.replace('空值', np.nan, inplace=True)
    reserved_cols = ['到院方式', '性别', '出生日期']
    reserved_data = data[reserved_cols].copy()
    data[['SBP', 'DBP']] = data['BP(mmHg)'].apply(lambda x: pd.Series(split_bp(x)))
    data.drop(columns=['BP(mmHg)'], inplace=True)
    numerical_cols = ['T℃', 'P(次/分)', 'R(次/分)', 'SpO2', 'SBP', 'DBP']
    data[numerical_cols] = data[numerical_cols].astype(float)
    initial_filled_data = data[numerical_cols].fillna(data[numerical_cols].mean())
    mask = ~data[numerical_cols].isna()
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(initial_filled_data)
    data_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    mask_tensor = torch.tensor(mask.values, dtype=torch.float32)
    return data_tensor, mask_tensor, scaler, numerical_cols, reserved_data


class DataImputer:
    def __init__(self, args, latent_dim=10, learning_rate=0.001, epochs=100, model_path='./weight/Imputation'):
        self.args = args
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.ep = epochs
        os.makedirs(model_path, exist_ok=True)
        self.model_path = os.path.join(model_path, str(args.rand_seed)+'_'+args.ImputMode + '_model.pkl')

    def Imupterselect(self):
        if self.args.ImputMode == 'MICE':        
            imputer = IterativeImputer(max_iter=10, random_state=42)
        elif self.args.ImputMode == 'RF':  
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                max_iter=10, random_state=42)
        elif self.args.ImputMode == 'VAE':
            imputer = VAEImputer(latent_dim=self.latent_dim, learning_rate=self.lr, epochs=self.ep)
        elif self.args.ImputMode == 'GAN':
            imputer = GANImputer(latent_dim=self.latent_dim, learning_rate=self.lr, epochs=self.ep)
        elif self.args.ImputMode == 'CGMM':
            imputer = CGMMImputer(n_components=self.args.n_comps)
        else:
            raise ValueError(f"Unknown method '{self.args.ImputMode}'. Supported: 'MICE', 'RF', 'VAE', 'GAN', 'CGMM'.")
        return imputer

    def impute(self, data, load_model=False):
        data.replace('空值', np.nan, inplace=True)
        if self.args.ImputMode in ['MICE', 'RF']:
            imputer = self.Imupterselect()

            numerical_cols = ['T℃', 'P(次/分)', 'R(次/分)', 'SpO2']
            data[numerical_cols] = data[numerical_cols].astype(float) 

            data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
            data[['SBP', 'DBP']] = data['BP(mmHg)'].apply(lambda x: pd.Series(split_bp(x)))
            bp_cols = ['SBP', 'DBP']
            data[bp_cols] = imputer.fit_transform(data[bp_cols])
            data['BP(mmHg)'] = data.apply(
                lambda row: f"{int(row['SBP'])}/{int(row['DBP'])}", axis=1)
            data['到院方式'] = data['到院方式'].fillna('未知')
            data.drop(columns=['SBP', 'DBP'], inplace=True)
            data = data.fillna('空值')
            return data

        elif self.args.ImputMode in ['GAN', 'VAE']:
            imputer = self.Imupterselect()
            if os.path.exists(self.model_path):
                imputer.load_model(self.model_path)
                completed_data = imputer.impute_missing_data(data)
            else:
                completed_data = imputer.impute_missing_data(data)
                imputer.save_model(self.model_path)
            return completed_data

        elif self.args.ImputMode == 'CGMM':
            imputer = self.Imupterselect()
            # if load_model and os.path.exists(self.model_path):
            if os.path.exists(self.model_path):
                imputer.load_model(self.model_path)
                completed_data = imputer.impute_missing_data(data)
            else:
                completed_data = imputer.impute_missing_data(data)
                imputer.save_model(self.model_path)
            return completed_data


# *****************************     CGMM      *********************************
class CGMMImputer:
    def __init__(self, n_components=3, max_iter=100, model_path='./weight/CGMM_model.pkl'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.model_path = model_path
        self.gmm = None 

    def fit_transform(self, data):
        data_tensor, mask_tensor, scaler, numerical_cols, reserved_data = preprocess_data(data)
        filled_data = data_tensor.numpy()
        self.gmm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter, random_state=42)
        self.gmm.fit(filled_data)
        
        missing_mask = np.isnan(filled_data)
        for col_idx in range(filled_data.shape[1]):
            if missing_mask[:, col_idx].any():
                observed_data = filled_data[~missing_mask[:, col_idx]]
                mu = self.gmm.means_[:, col_idx]
                sigma = self.gmm.covariances_[:, col_idx, col_idx]
                cross_sigma = self.gmm.covariances_[:, col_idx, :]

                cond_mean = []
                cond_cov = []
                for k in range(self.n_components):
                    mean_k = mu[k] + cross_sigma[k].dot(
                        np.linalg.inv(self.gmm.covariances_[k][:, :]).dot(
                            (observed_data - self.gmm.means_[k]).T).T
                    )
                    cond_mean.append(mean_k)
                    cov_k = sigma[k] - cross_sigma[k].dot(
                        np.linalg.inv(self.gmm.covariances_[k][:, :]).dot(cross_sigma[k].T)
                    )
                    cond_cov.append(cov_k)

                for j, idx in enumerate(np.where(missing_mask[:, col_idx])[0]):
                    component = np.random.choice(self.n_components, p=self.gmm.weights_)
                    sampled_value = np.random.normal(cond_mean[component][j], np.sqrt(cond_cov[component]))
                    filled_data[idx, col_idx] = sampled_value
        
        filled_data = scaler.inverse_transform(filled_data)
        filled_df = pd.DataFrame(filled_data, columns=numerical_cols)

        filled_df = pd.concat([reserved_data.reset_index(drop=True), filled_df], axis=1)
        filled_df['BP(mmHg)'] = filled_df.apply(lambda row: f"{int(row['SBP'])}/{int(row['DBP'])}", axis=1)
        filled_df.drop(columns=['SBP', 'DBP'], inplace=True)
        filled_df = filled_df.fillna('空值')
        return filled_df

    def impute_missing_data(self, data):
        return self.fit_transform(data)

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        with open(path, 'wb') as f:
            pickle.dump(self.gmm, f)

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        with open(path, 'rb') as f:
            self.gmm = pickle.load(f)


# *****************************     VAE      *********************************
class VAEImputer:
    def __init__(self, latent_dim=10, learning_rate=0.001, epochs=100):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scaler = MinMaxScaler()

    def generate_noise(self, batch_size, input_dim):
        return torch.randn(batch_size, input_dim)

    class VAE(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super(VAEImputer.VAE, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 64),nn.ReLU(),
                                         nn.Linear(64, 32),nn.ReLU(),nn.Linear(32, latent_dim * 2))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 32),nn.ReLU(),
                                         nn.Linear(32, 64),nn.ReLU(),nn.Linear(64, input_dim))
        def forward(self, x):
            q = self.encoder(x)
            mean, log_var = torch.chunk(q, 2, dim=1)
            std = torch.exp(0.5 * log_var)
            z = mean + std * torch.randn_like(std)
            x_reconstructed = self.decoder(z)
            return x_reconstructed, mean, log_var

    def vae_loss(self, x, x_reconstructed, mean, log_var, mask):
        device = x.device
        x_reconstructed = x_reconstructed.to(device)
        mask = mask.to(device)
    
        reconstruction_loss = torch.sum(mask * (x - x_reconstructed) ** 2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        return reconstruction_loss + kl_loss

    def train_module(self, data_tensor, mask_tensor, input_dim):
        vae = self.VAE(input_dim, self.latent_dim).to(device)  
        vae = vae.to(device)  
        data_tensor = data_tensor.to(device)  
        
        optimizer = optim.Adam(vae.parameters(), lr=self.learning_rate)
        best_loss = float('inf')
        epochs_no_improve = 0
        patience = 10  

        for epoch in range(self.epochs):
            vae.train()
            x_reconstructed, mean, log_var = vae(data_tensor)  
            loss = self.vae_loss(data_tensor, x_reconstructed, mean, log_var, 1 - mask_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        vae.eval()
        with torch.no_grad():
            completed_data, _, _ = vae(data_tensor)
        return completed_data.squeeze().cpu().numpy() 

    def impute_missing_data(self, data):
        data_tensor, mask_tensor, scaler, numerical_cols, reserved_data = preprocess_data(data)
        completed_data = self.train_module(data_tensor, mask_tensor, data_tensor.size(1))
        filled_data = scaler.inverse_transform(completed_data)
        filled_df = pd.DataFrame(filled_data, columns=numerical_cols)

        filled_df = pd.concat([reserved_data.reset_index(drop=True), filled_df], axis=1)
        filled_df['BP(mmHg)'] = filled_df.apply(lambda row: f"{int(row['SBP'])}/{int(row['DBP'])}", axis=1)
        filled_df.drop(columns=['SBP', 'DBP'], inplace=True)
        filled_df = filled_df.fillna('空值')
        return filled_df

    def save_model(self, path):
        torch.save(self, path)

    def load_model(self, path):
        model = torch.load(path)
        self.__dict__.update(model.__dict__)

# *****************************     GAN      *********************************
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))
    def forward(self, x):
        return self.model(x)
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    def forward(self, x):
        return self.model(x)

class GANImputer:
    def __init__(self, latent_dim=10, learning_rate=0.001, epochs=100):
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.scaler = MinMaxScaler()

    def generate_noise(self, batch_size, input_dim):
        return torch.randn(batch_size, input_dim)

    def train_module(self, data_tensor, mask_tensor, input_dim, hidden_dim=64, patience=10, min_delta=1e-4):
        data_tensor = data_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        batch_size = data_tensor.size(0)
        
        generator = Generator(input_dim, hidden_dim).to(device)
        discriminator = Discriminator(input_dim, hidden_dim).to(device)
        
        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(generator.parameters(), lr=self.lr)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=self.lr)
        
        best_loss_g = float('inf')
        epochs_no_improve = 0
        best_generator_state = None
        
        for epoch in range(self.epochs):
            real_data = data_tensor
            noise = self.generate_noise(batch_size, input_dim).to(device)
            fake_data = generator(noise) * (1 - mask_tensor) + data_tensor * mask_tensor
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data.detach())
            
            loss_d_real = criterion(real_output, real_labels)
            loss_d_fake = criterion(fake_output, fake_labels)
            loss_d = loss_d_real + loss_d_fake
            
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            
            fake_output = discriminator(fake_data)
            loss_g = criterion(fake_output, real_labels) + torch.mean((fake_data - data_tensor) ** 2 * (1 - mask_tensor))
            
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            if loss_g.item() < best_loss_g - min_delta:
                best_loss_g = loss_g.item()
                epochs_no_improve = 0
                best_generator_state = generator.state_dict() 
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss G: {loss_g.item():.4f}, Loss D: {loss_d.item():.4f}")
        if best_generator_state is not None:
            generator.load_state_dict(best_generator_state)
            print("Restored generator to the best state.")

        return generator.cpu()  # 将生成器移回CPU

    def impute_missing_data(self, data):
        data_tensor, mask_tensor, scaler, numerical_cols, reserved_data = preprocess_data(data)
        generator = self.train_module(data_tensor, mask_tensor, input_dim=data_tensor.size(1))

        with torch.no_grad():
            fake_data = generator(data_tensor).numpy()
            filled_data = fake_data * (1 - mask_tensor.numpy()) + data_tensor.numpy() * mask_tensor.numpy()
        filled_data = scaler.inverse_transform(filled_data)
        filled_df = pd.DataFrame(filled_data, columns=numerical_cols)

        filled_df = pd.concat([reserved_data.reset_index(drop=True), filled_df], axis=1)
        filled_df['BP(mmHg)'] = filled_df.apply(lambda row: f"{int(row['SBP'])}/{int(row['DBP'])}", axis=1)
        filled_df.drop(columns=['SBP', 'DBP'], inplace=True)
        filled_df = filled_df.fillna('空值')
        return filled_df

    def save_model(self, path):
        torch.save(self, path)

    def load_model(self, path):
        model = torch.load(path)
        self.__dict__.update(model.__dict__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Triage")
    parser.add_argument("--ImputMode", default='VAE', 
                        help="'RF', 'GAN', 'MICE', 'VAE'")
    args = parser.parse_args(args = [])

    data = pd.DataFrame({
        '到院方式': ['步入', '空值', '步入', '轮椅', '120'],
        '性别': ['男', '女', '女', '男', '女'],
        '出生日期': ['1990-5-9', '1985-7-8', '1993-5-9', '1950-6-9', '空值'],
        'T℃': [36.2, 36.5, 36.2, 36.5, 36.0],
        'P(次/分)': ['空值', 100, '空值', 68, 91],
        'R(次/分)': ['空值', 15, '空值', 18, 19],
        'BP(mmHg)': ['空值', '/85', '空值', '148/', '191/121'],
        'SpO2': ['99', '96', '99', '95', '92']
    })

    imputer = DataImputer(args)
    imputed_data = imputer.impute(data)
    print(imputed_data)
