from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import numpy as np
import webbrowser

import sys
sys.path.append('angular_spectrum/')
import sim.propagate as prop
# Usage: d2nn.d2nn_inference(X), returns X. 
# You may also want to show the frame which the probability of each class is computed on.
# therefore, you can use d2nn.draw_frame
# Then you can set only_return_X = False to get (X, y_pred)
import sim.d2nn_inference as d2nn
from util import get_obj

app = Flask(__name__, static_folder='static', template_folder='templates')

d2nn_or_prop = 'd2nn'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation(d2nn_or_prop=d2nn_or_prop):

    data = request.get_json()
    z = float(data.get('z'))
    wl = float(data.get('wl'))/1000
    grid = float(data.get('grid'))
    matrix = torch.Tensor(np.array(data.get('matrix')))
    types = data.get('types')
    n = matrix.shape[0]
    
    if d2nn_or_prop == 'prop':
        X = prop.propagate(matrix, grid, wl, z)
    elif d2nn_or_prop == 'd2nn':
        # NOTE: My understanding of the implementation is probably wrong. Just
        # delete these lines. They are just a demonstration how you can use the 
        # function.
        # The reason why this does not work is that
        # `matrix` is not of the correct size of 28 * 28.
        
        X, y_pred = d2nn.d2nn_inference(matrix, only_return_X=False)

    X = X.abs()

    return jsonify({"result":X.tolist()})

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8080")
    app.run(debug=False, port=8080)
    
