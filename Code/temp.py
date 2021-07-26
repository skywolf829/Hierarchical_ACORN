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
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    total_train_time = [9 + 7/60, 
                        7 + 21/60, 
                        6 + 44/60, 
                        6 + 11/60, 
                        6 + 1/60,
                        5 + 59/60,
                        5 + 45/60, 
                        5 + 46/60]

    model1_train_time = [25/60, 
                        26/60, 
                        27/60, 
                        26/60, 
                        28/60,
                        28/60,
                        28/60, 
                        28/60]

    model2_train_time = [56/60, 
                        59/60,
                        1,
                        59/60,
                        1 + 0/60,
                        1 + 1/60,
                        1 + 0/60, 
                        1 + 1/60]

    model3_train_time = [1 + 37/60, 
                        1 + 38/60,
                        1 + 40/60, 
                        1 + 38/60,
                        1 + 40/60,
                        1 + 39/60,
                        1 + 40/60, 
                        1 + 39/60]

    model4_train_time = [2 + 38/60, 
                        2 + 35/60,
                        2 + 34/60,
                        2 + 28/60, 
                        2 + 30/60, 
                        2 + 29/60, 
                        2 + 28/60,
                        2 + 28/60]

    model5_train_time = [4 + 31/60, 
                        4 + 6/60,
                        3 + 53/60,
                        3 + 40/60, 
                        3 + 40/60,
                        3 + 40/60,
                        3 + 35/60,
                        3 + 36/60]    

    model6_train_time = [9 + 7/60, 
                        7 + 21/60, 
                        6 + 44/60, 
                        6 + 11/60, 
                        6 + 1/60,
                        5 + 59/60,
                        5 + 45/60, 
                        5 + 46/60]  
    
    for i in range(len(model5_train_time)):
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
    total_train_time = np.array(total_train_time)

    plt.plot(x, total_train_time, marker='^')
    plt.title("Total training time")
    plt.ylabel("Time (minutes)")
    plt.xlabel("# GPUs")

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.ylim(bottom=0)

    plt.show()

    plt.clf()
    plt.bar(x, model1_train_time, color='r', label='Model 0')
    plt.bar(x, model2_train_time, bottom=model1_train_time, color='b', label='Model 1')
    plt.bar(x, model3_train_time, bottom=model2_train_time+model1_train_time, color='g', label='Model 2')
    plt.bar(x, model4_train_time, bottom=model3_train_time+model2_train_time+model1_train_time, color='y', label='Model 3')
    plt.bar(x, model5_train_time, bottom=model4_train_time+model3_train_time+model2_train_time+model1_train_time, color='purple', label='Model 4')
    plt.bar(x, model6_train_time, bottom=model5_train_time+model4_train_time+model3_train_time+model2_train_time+model1_train_time, color='gray', label='Model 5')
    plt.ylabel("Time (minutes)")
    plt.xlabel("# GPUs")

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.legend()
    plt.title("Training time per model")
    plt.show()

if __name__ == '__main__':
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    f = h5py.File("./TrainingData/channel_ts0.h5", 'r')
    print(f['data'].shape)

