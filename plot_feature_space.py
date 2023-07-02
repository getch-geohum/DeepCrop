# Import necessary packages to the environment
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from tqdm import tqdm
import sys
import os
import copy
import random
import math
import itertools
import pickle
import ast, csv
from functools import partial
import warnings
import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage import measure
from skimage.io import imread
from glob import glob
from time import gmtime, strftime
from scipy.spatial import distance
from skimage.io import imsave
import string  
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm # new
from sklearn.manifold import TSNE

import seaborn as sns
import torchvision.models as models

from ipdb import set_trace as st
import torch
import argparse
import json
import os
import copy
import itertools
from skimage.io import imread
from skimage import measure
from skimage import draw


class GenerateLatentSpace:
    def __init__(self, data_root=None, out_root=None, model_name=None, data=None):
        self.data_root = f'{data_root}/{model_name}'
        self.out_root = out_root
        self.model_name = model_name
        self.data = data
        self.format = '.npy'
        
    def reduceDime(self):
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root, exist_ok=True)
        folds = os.listdir(self.data_root)        
        
        all_files = glob(f'{self.data_root}/*.npy')
        features = sorted([file for file in all_files if not 'label' in file])
        labels = sorted([file for file in all_files if 'label' in file])
        #print(features)
        assert len(features) == len(labels), 'features and labels are note the samae'
        
        arrs = np.vstack([np.load(feat) for feat in features])
        lbls = np.concatenate([np.argmax(np.load(lbl),axis=-1).ravel() for lbl in labels])
        
        assert len(arrs) == len(lbls), f'Features and labels are not the same length, {arrs.shape} and {lbls.shape} respectively'''
        print(len(arrs), len(lbls))

        tsne = TSNE(n_components=2,
                    init='pca',
                    perplexity=50,
                    n_iter=500,
                    n_jobs=-1).fit_transform(arrs)
        
        crop_names = ['Guizota','Maize','Millet','Others','Pepper','Teff']
        color_names = ['red', 'blue', 'green', 'yellow', 'orange', 'pink']
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        for i in range(len(crop_names)):
            sub = tsne[lbls == i]
            ax.scatter(sub[:,0], sub[:,1], label=crop_names[i], s=1.2, alpha=0.9, color = color_names[i])
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=10, bbox_to_anchor=(0.5,0.95)) 
        plt.savefig(f'{self.out_root}/feat_{self.model_name}_{self.data}.png', dpi=350, bbox_inches='tight')
        plt.show()

        
def argumentParser():
    parser = argparse.ArgumentParser(description = 'Deep feature space embeding plot')
    parser.add_argument('--data_root', help='data folder', type=str, required=False, default='/home/getch/ssl/EO_Africa/FEATURES')
    parser.add_argument('--out_root', help = 'main root to save the test result', type = str, required=False, default='/home/getch/ssl/EO_Africa/PLOTS')
    parser.add_argument('--model_name', help = 'main root to save the test result', type = str, required=False, default='tempCNN')
    parser.add_argument('--data', help = 'The data name to which to plot', type = str, required=False, default='sentinel')
    arg = parser.parse_args()
    return arg

if __name__ == "__main__":
    args = argumentParser()
    generator = GenerateLatentSpace(data_root=args.data_root,
                                    out_root=args.out_root,
                                    model_name=args.model_name,
                                    data=args.data)
    generator.reduceDime()


