from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import webbrowser

import sys
sys.path.append('angular_spectrum/')

import sim.propagate as prop
from util import get_obj


plt.switch_backend('Agg')

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')  # Make sure your HTML file is in a 'templates' folder

@app.route('/run_simulation', methods=['POST'])
def run_simulation():

    data = request.get_json()
    z = float(data.get('z'))
    wl = float(data.get('wl'))
    grid = float(data.get('grid'))
    matrix = torch.Tensor(np.array(data.get('matrix')))
    types = data.get('types')
    n=matrix.shape[0]

    all_elements_are_zero = all(all(element == 0 for element in row) for row in matrix)
    if all_elements_are_zero:
        if types == "Circle":
            matrix = get_obj.obj(get_obj.circle, n)
        if types == "Double slit":
            matrix = get_obj.obj(get_obj.double_slit,n)
        if types == "Square":
            matrix = get_obj.obj(get_obj.square,n)

    
    X = prop.propagate(matrix, grid, wl, z)
    #X = prop.propagate(get_obj(get_double_slit), grid, wl, z)

    #show_mat(get_obj(get_double_slit), grid)
    get_obj.show_mat(matrix, grid)
    plt.savefig(os.path.join(app.static_folder, 'start_plot.png'))
    plt.close()

    get_obj.show_mat(X, grid)
    plt.savefig(os.path.join(app.static_folder, 'simulation_plot.png'))
    plt.close()

    image_path = ""

    return jsonify({'image_path': image_path})

@app.route('/static/<filename>')
def serve_image(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8080")
    app.run(debug=False, port=8080)
    
