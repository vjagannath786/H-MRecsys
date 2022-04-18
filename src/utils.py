import os
import numpy as np
import pandas as pd
import pickle


def save_txt(data_to_save, filepath, mode='a'):
    """
    Save text to a file.
    """
    with open(filepath, mode) as text_file:
        text_file.write(data_to_save + '\n')




def save_outputs(files_to_save: dict,
                 folder_path):
    """
    Save objects as pickle files, in a given folder.
    """
    for name, file in files_to_save.items():
        with open(folder_path + name + '.pkl', 'wb') as f:
            pickle.dump(file, f)



def softmax(x):
    """
    (Currently not used.) Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()