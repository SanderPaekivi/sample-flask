#import os
#import http.server
#import socketserver
#
#from http import HTTPStatus
#
#
#class Handler(http.server.SimpleHTTPRequestHandler):
#    def do_GET(self):
#        self.send_response(HTTPStatus.OK)
#        self.end_headers()
#        msg = 'Hello! you requested %s' % (self.path)
#        self.wfile.write(msg.encode())
#
#
#port = int(os.getenv('PORT', 80))
#print('Listening on port %s' % (port))
#httpd = socketserver.TCPServer(('', port), Handler)
#httpd.serve_forever()




import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy.stats import levy_stable, norm
import torch
import torch.nn as nn
import pickle
import flask
from flask import Flask, render_template, request, send_file 
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64



# Running the flask app
#app = Flask(__name__)

####################################
### How to locate template file? ###
####################################
app = Flask(__name__)
#run_with_ngrok(app)

@app.route('/', methods=['GET'])
def home():
    
    return render_template('index.html', prediction_text = '')
    #return 'hello world'

@app.route('/', methods=['POST'])
def predict():
    #alpha_to_gen = list(request.form.values())
    #print(alpha_to_gen)
    #alpha_to_gen = float(alpha_to_gen[0])
    #ts_len = float(alpha_to_hen[1])

    inputs = request.form.to_dict()
    print(inputs)
    alpha_to_gen = float(inputs['Alpha'])
    ts_len = int(inputs['tsl'])

    #ts_len = 250
    
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
        
    model = LSTMnetwork()
    model = torch.load('LSTM100x1_model.pkl')

    base = __ma_model(ts_len)
    frac_base = __frac_diff(base, alpha_to_gen)
    data = __arma_model(frac_base)

    with torch.no_grad():
      model.eval()
      
      seq = torch.FloatTensor(data)
      with torch.no_grad():
          model.hidden = (torch.zeros(1,1,model.hidden_size),
                          torch.zeros(1,1,model.hidden_size))
          
          prediction = round(model(seq).item(),2)


    img = BytesIO()
    plt.plot(data)
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')


    return render_template('index.html', prediction_text='The Hurst parameter of generated data was estimated by the LSTM to be {}'.format(prediction), 
                           figure_to_print = plot_url)


