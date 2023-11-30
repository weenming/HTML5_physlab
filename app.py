from flask import Flask, request, jsonify, render_template, send_from_directory

import sys
sys.path.append('angular_spectrum/')

import sim.propagate as prop
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import webbrowser

plt.switch_backend('Agg')

app = Flask(__name__, static_folder='static', template_folder='templates')

def get_circle(halfn):
    r = 10
    X = torch.arange(-halfn, halfn) ** 2 + torch.arange(-halfn, halfn).unsqueeze(-1) ** 2
    idx = X < r ** 2
    n_idx = X >= r ** 2

    X[n_idx] = 0
    X[idx] = 1
    return X

def get_square(halfn):
    s = 10
    idx = torch.arange(halfn - s, halfn + s)
    idx = (idx.repeat(2 * s), 
        idx.repeat_interleave(2 * s))
    

    X = torch.zeros((halfn * 2, halfn * 2))
    X[idx] = 1
    return X

def get_double_slit(halfn):
    w = 10
    d = 2
    x_left_idx = torch.arange(halfn - w // 2 - d // 2, halfn - w // 2 + d // 2)
    x_right_idx = torch.arange(halfn + w // 2 - d // 2, halfn + w // 2 + d // 2)
    x_left_idx = x_left_idx.repeat(halfn * 2)
    x_right_idx = x_right_idx.repeat(halfn * 2)
    y_idx = torch.arange(0, halfn * 2).repeat_interleave(d)
      
    
    X = torch.zeros((halfn * 2, halfn * 2))
    X[x_left_idx, y_idx] = 1
    X[x_right_idx, y_idx] = 1
    return X

def get_obj(get_x=get_circle):
    n = 256 # <~10000
    halfn = n // 2
    X = get_x(halfn)

    X = X / X.max() # normalize
    X = X.to(torch.complex128) # NOTE: complex 64 induces accumulated error which is catastrophic
    return X

# preprocess

def cut(X, show_ratio):
    halfn = X.shape[0] // 2
    lower = int(halfn * (1 - show_ratio))
    higher = int(halfn * (1 + show_ratio))
    
    return X[lower: higher, lower:higher]

def show_mat(X, pixel, show_ratio=1):
    X = cut(X, show_ratio)
    halfn = X.shape[0] // 2
    
    X = X.abs().cpu().numpy()

    fig, ax = plt.subplots(1, 1)
    s = ax.imshow(
        X, 
        extent=[-halfn * pixel, halfn * pixel, -halfn * pixel, halfn * pixel]
    )
    ax.set_xlabel('$\mu$m')
    ax.set_ylabel('$\mu$m')
    ax.set_title('objective')
    fig.colorbar(s)

    return


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

    all_elements_are_zero = all(all(element == 0 for element in row) for row in matrix)
    if all_elements_are_zero:
        if types == "Circle":
            matrix = get_obj(get_circle)
        if types == "Double slit":
            matrix = get_obj(get_double_slit)
        if types == "Square":
            matrix = get_obj(get_square)

    
    X = prop.propagate(matrix, grid, wl, z)
    #X = prop.propagate(get_obj(get_double_slit), grid, wl, z)

    #show_mat(get_obj(get_double_slit), grid)
    show_mat(matrix, grid)
    plt.savefig(os.path.join(app.static_folder, 'start_plot.png'))
    plt.close()

    show_mat(X, grid)
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
    
