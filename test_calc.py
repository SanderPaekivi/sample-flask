import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy.stats import levy_stable, norm


def __ma_model(n_points, noise_std = 1, noise_alpha = 2, params = []):
    ma_order = len(params)
    if noise_alpha == 2:
        noise = norm.rvs(scale=noise_std, size=(n_points + ma_order))
    else:
        noise = levy_stable.rvs(
            noise_alpha, 0, scale=noise_std, size=(n_points + ma_order)
        )
    
    if ma_order == 0:
        return noise
    ma_coeffs = np.append([1], params)
    ma_series = np.zeros(n_points)
    for idx in range(ma_order, n_points + ma_order):
        take_idx = np.arange(idx, idx - ma_order - 1, -1).astype(int)
        ma_series[idx - ma_order] = np.dot(ma_coeffs, noise[take_idx])
    return ma_series[ma_order:]

def __frac_diff(x, d):
    
    def next_pow2(n):
        return (n - 1).bit_length()

    n_points = len(x)
    fft_len = 2 ** next_pow2(2 * n_points - 1)
    prod_ids = np.arange(1, n_points)
    frac_diff_coefs = np.append([1], np.cumprod((prod_ids - d - 1) / prod_ids))
    dx = ifft(fft(x, fft_len) * fft(frac_diff_coefs, fft_len))
    return np.real(dx[0:n_points])

def __arma_model(noise, params = []):

    ar_order = len(params)
    if ar_order == 0:
        return noise
    n_points = len(noise)
    arma_series = np.zeros(n_points + ar_order)
    for idx in np.arange(ar_order, len(arma_series)):
        take_idx = np.arange(idx - 1, idx - ar_order - 1, -1).astype(int)
        arma_series[idx] = np.dot(params, arma_series[take_idx]) + noise[idx - ar_order]
    return arma_series[ar_order:]

arfima_set_N = 2000
test_set_percentage = 0.3

test_set_len = int(arfima_set_N*0.3)

ts_len = 200
alphas = [i-0.5 for i in np.random.rand(arfima_set_N)]

arfima_data = []

for alpha in alphas:
  base = __ma_model(ts_len)
  frac_base = __frac_diff(base, alpha)

  arfima_data.append((__arma_model(frac_base),alpha))

train_data = arfima_data[0:len(arfima_data)-test_set_len]
test_data = arfima_data[len(arfima_data)-test_set_len:]

#torch.manual_seed(8888)

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle

#saving the model
#pickle.dump(model, open('/content/drive/My Drive/Colab Notebooks/Science/Transient_Hurst/LSTM100x1_ArfimaAlpha.pkl', 'wb'))

class LSTMnetwork(nn.Module):
    def __init__(self,input_size=1,hidden_size=100,output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Add an LSTM layer:
        self.lstm = nn.LSTM(input_size,hidden_size)
        
        # Add a fully-connected layer:
        self.linear = nn.Linear(hidden_size,output_size)
        
        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))

    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq),1,-1), self.hidden)
        #return lstm_out
        pred = self.linear(lstm_out.view(len(seq),-1))
        #pred = self.linear(lstm_out[-50:].view(50,-1))
        return pred[-1]
    
#torch.manual_seed(101)
#model = LSTMnetwork()

model = pickle.load(open('LSTM100x1_ArfimaAlpha.pkl','rb'))

model.hidden = (torch.zeros(1,1,model.hidden_size),
                       torch.zeros(1,1,model.hidden_size))

seq = torch.FloatTensor(test_data[56][0]).view(-1)

print(test_data[56][1])
print(model( seq ))#.view(len(seq),-1)[-1]

