import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = 200, z_dim = 20):
        super().__init__()
        
        #encoder
        self.im_2hid = nn.Linear(input_dim, hidden_dim)
        self.hid_2mu = nn.Linear(hidden_dim, z_dim)
        self.hid_2sigma = nn.Linear(hidden_dim, z_dim)
        
        #decoder
        self.z_2hid = nn.Linear(z_dim, hidden_dim)
        self.hid_2img = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU(inplace=True)
        
        
    def encode(self, x):
        #q_phi(z|x)
        h = self.relu(self.im_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
    
    def decode(self, z):
        #p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparameterized)
        return x_reconstructed, mu, sigma
    
if __name__ == "__main__":
    x = torch.randn(4, 784)
    vae = VariationalAutoEncoder(input_dim=784)
    x_new, mu_new, sigma_new = vae.forward(x)
    