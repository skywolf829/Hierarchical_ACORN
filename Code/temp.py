from torch.autograd.grad_mode import F
from utility_functions import str2bool
from options import *
import argparse
import os
import h5py
import imageio
import numpy as np
from models import PositionalEncoding
import torch
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from utility_functions import make_coord
from torch.linalg import norm
from matplotlib.ticker import ScalarFormatter

def create_toy_data():
    f = h5py.File("./3D_toydata.h5", 'w')
    coords = make_coord([128, 128, 128], "cuda:0", flatten=False)
    coords = norm(coords, dim=3) * 5
    coords = coords.unsqueeze(0).cpu().numpy()
    coords = np.sinc(coords)

    coords -= coords.min()
    coords *= (1/coords.max())

    f['data'] = coords
    f.close()

def create_graphs():
    x = [1, 2, 4, 8]
    total_train_time = [65 + 29/60,
    39 + 55/60,
    28 + 37/60,
    21 + 50/60]

    model1_train_time = [1 + 42/60,
    1 + 42/60,
    1 + 42/60,
    1 + 44/60]

    model2_train_time = [3 + 50/60,
    3 + 13/60,
    3 + 3/60,
    3 + 7/60]

    model3_train_time = [6 + 20/60,
    5 + 1/60,
    4 + 40/60,
    4 + 42/60]

    model4_train_time = [9 + 36/60,
    7 + 17/60,
    6 + 40/60,
    6 + 32/60]

    model5_train_time = [15 + 27/60,
    10 + 56/60,
    9 + 33/60,
    8 + 57/60]    

    model6_train_time = [28 + 53/60,
    18 + 47/60,
    14 + 44/60,
    12 + 52/60]  

    model7_train_time = [65 + 29/60,
    39 + 55/60,
    28 + 37/60,
    21 + 50/60] 
    
    for i in range(len(model7_train_time)):
        model7_train_time[i] -= model6_train_time[i]
        model6_train_time[i] -= model5_train_time[i]
        model5_train_time[i] -= model4_train_time[i]
        model4_train_time[i] -= model3_train_time[i]
        model3_train_time[i] -= model2_train_time[i]
        model2_train_time[i] -= model1_train_time[i]
        

    model1_train_time = np.array(model1_train_time)
    model2_train_time = np.array(model2_train_time)
    model3_train_time = np.array(model3_train_time)
    model4_train_time = np.array(model4_train_time)
    model5_train_time = np.array(model5_train_time)
    model6_train_time = np.array(model6_train_time)
    model7_train_time = np.array(model7_train_time)
    total_train_time = np.array(total_train_time)

    fig, ax = plt.subplots()

    ax.plot(x, model7_train_time, marker='^', label='Actual')
    ax.set_title("Model 6 train time")
    ax.set_ylabel("Time (minutes)")
    ax.set_xlabel("# GPUs")

    ideal = [model7_train_time[0]]
    for i in range(1, len(total_train_time)):
        ideal.append((ideal[0]/ (2**i)))

    ax.plot(x, ideal, '--', label='ideal')

    #ax.set_xticks([1, 2, 4, 8], ['1', '2', '4', '8'])
    #plt.ylim(bottom=0)
    ax.loglog()
    ax.legend()
    
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_powerlimits((-5, 5))
        axis.set_major_formatter(formatter)
        axis.set_minor_formatter(formatter)
    plt.show()

    plt.clf()

    fig, ax = plt.subplots()
    x = [1, 2, 3, 4]
    ax.bar(x, model1_train_time, color='r', label='Model 0')
    ax.bar(x, model2_train_time, bottom=model1_train_time, color='b', label='Model 1')
    ax.bar(x, model3_train_time, bottom=model2_train_time+model1_train_time, color='g', label='Model 2')
    ax.bar(x, model4_train_time, bottom=model3_train_time+model2_train_time+model1_train_time, color='y', label='Model 3')
    ax.bar(x, model5_train_time, bottom=model4_train_time+model3_train_time+model2_train_time+model1_train_time, color='purple', label='Model 4')
    ax.bar(x, model6_train_time, bottom=model5_train_time+model4_train_time+model3_train_time+model2_train_time+model1_train_time, color='gray', label='Model 5')
    ax.bar(x, model7_train_time, bottom=model6_train_time+model5_train_time+model4_train_time+model3_train_time+model2_train_time+model1_train_time, color='orange', label='Model 6')

    ax.set_ylabel("Time (minutes)")
    ax.set_xlabel("# GPUs")
    plt.xticks([1, 2, 3, 4], ['1', '2', '4', '8'])
    ax.legend()
    ax.set_title("Training time per model")
    plt.show()

if __name__ == '__main__':
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    create_graphs()

