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

if __name__ == '__main__':
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")

    x = [1, 2, 3, 4, 5, 6, 7, 8]
    total_train_time = [20 + 52/60, 
                        12 + 38/60, 
                        11 + 28/60, 
                        8 + 53/60, 
                        8 + 31/60,
                        8 + 22/60,
                        7 + 58/60, 
                        7 + 58/60]

    model1_train_time = [30/60, 
                        31/60, 
                        31/60, 
                        32/60, 
                        32/60,
                        32/60,
                        31/60, 
                        31/60]

    model2_train_time = [1 + 11/60, 
                        1 + 13/60,
                        1 + 13/60,
                        1 + 14/60,
                        1 + 16/60,
                        1 + 16/60,
                        1 + 14/60, 
                        1 + 16/60]

    model3_train_time = [2 + 6/60, 
                        2 + 5/60,
                        2 + 6/60, 
                        2 + 7/60,
                        2 + 8/60,
                        2 + 7/60,
                        2 + 5/60, 
                        2 + 6/60]

    model4_train_time = [3 + 37/60, 
                        3 + 23/60,
                        3 + 22/60,
                        3 + 16/60, 
                        3 + 18/60, 
                        3 + 17/60, 
                        3 + 12/60,
                        3 + 14/60]

    model5_train_time = [7 + 42/60, 
                        5 + 48/60,
                        5 + 26/60,
                        5 + 6/60, 
                        5 + 4/60,
                        5 + 2/60,
                        4 + 53/60,
                        4 + 53/60]    

    model6_train_time = [20 + 52/60, 
                        12 + 38/60, 
                        11 + 28/60, 
                        8 + 53/60, 
                        8 + 31/60,
                        8 + 22/60,
                        7 + 58/60, 
                        7 + 58/60]   
    
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