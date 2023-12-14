from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import webbrowser

import sys
sys.path.append('angular_spectrum/')
import sim.propagate as prop

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mnist')
def mnist():
    return render_template('MNIST.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.get_json()
    z = float(data.get('z'))
    wl = float(data.get('wl'))/1000
    grid = float(data.get('grid'))
    matrix = torch.Tensor(np.array(data.get('matrix')))
    
    X = prop.propagate(matrix, grid, wl, z)
    X = X.abs()

    return jsonify({"result":X.tolist()})

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    z = float(data.get('z'))
    wl = float(data.get('wl'))/1000
    grid = float(data.get('grid'))
    matrix = torch.Tensor(np.array(data.get('matrix')))
    
    X = prop.propagate(matrix, grid, wl, z)
    X = X.abs()

    return jsonify(X.tolist())

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8080")
    app.run(debug=False, port=8080)
    
