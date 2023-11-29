from flask import Flask, request, jsonify, render_template, send_from_directory

import sys
#sys.path.append('./../../')

#import angular_spectrum.sim.propagate as prop
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

plt.switch_backend('Agg')

app = Flask(__name__, static_folder='static', template_folder='templates')

def perform_simulation(z):
    # Replace this with your actual simulation logic using Matplotlib
    x = np.linspace(0, 10, 100)
    y = np.sin(z * x)

    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Simulation Plot')
    
    # Save the plot as an image in the static folder
    
    plt.savefig(os.path.join(app.static_folder, 'simulation_plot.png'))
    plt.close()  # Close the plot to free up resources

    return 'static/simulation_plot.png'


@app.route('/')
def index():
    return render_template('index.html')  # Make sure your HTML file is in a 'templates' folder

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    # Get input parameters from the request
    data = request.get_json()
    z = float(data.get('z'))

    # Perform the optical simulation and get the path of the saved plot
    image_path = perform_simulation(z)

    # Return the path of the saved image to the web page
    return jsonify({'image_path': image_path})

@app.route('/static/<filename>')
def serve_image(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
