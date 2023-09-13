import pandas as pd
import random
import pickle
import numpy as np
import scipy.stats
import argparse
import os
import vcf
import h5py
import scipy.stats

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from os.path import exists

os.getcwd()
os.chdir("./")
opath="./output/"
perc_mask = 50

output_file = "test_output_cnn_imputed_%d" % (perc_mask) 

num_epoch=3
group_size=20
lr=0.000100
kr=0.000001
drop_perc=0.000000
n_layers=5
gamma=5
optimizer="adam"
hidden_layer_s=32

window_size=20
cnn_out = opath+output_file+'/output_cnn_epoch%d_group%d_lr%f_kr%f_drop%f_layers%d_gamma%d_optimizer%s_hidden%d_local.txt' % (num_epoch, group_size, lr, kr, drop_perc, n_layers, gamma, optimizer, hidden_layer_s)
cnn_pred = pd.read_csv(cnn_out)


beagle_out = "./beagle_data/processed/beagle_out_hap_df_ref_%d.pkl" % (perc_mask)
true_out = "./beagle_data/processed/test_eval_hap_df.pkl"


filehandler = open(beagle_out,"rb")
beagle_pred = pickle.load(filehandler)
filehandler.close()

# true = pd.read_csv(true_out, delimiter="\t")
filehandler = open(true_out,"rb")
true = pickle.load(filehandler)
filehandler.close()


### USE THRESHOLD TO CONVERT FLOAT PREDICTIONS TO BINARY PREDICTIONS
thrd = 0.3
cnn_pred_bin = cnn_pred.applymap(lambda x: 0 if x > thrd else 1)

### WITH BINARY INTEGERS, JUST USE THE PREDICTIONS
cnn_pred_bin = cnn_pred

test_input_df = pd.read_csv("./beagle_data/processed/test_input_%d.hap" % (perc_mask), delimiter="\t")
num_col = test_input_df.shape[1]
test_input_df_ss = test_input_df.iloc[:,1:num_col]
test_input_df_ss = test_input_df_ss.T

### Subset to number of SNPs that are imputed
num_cnn_snps = cnn_pred.shape[0]
beagle_pred = beagle_pred.iloc[0:num_cnn_snps,:]
true_ss = true.iloc[0:num_cnn_snps,:]
test_input_df_ss = test_input_df_ss.iloc[0:num_cnn_snps,:]

## Evaluate R^2 across individuals
# r_2_l_b = []
r_2_l_1 = []
r_2_l_2 = []

for snp in range(test_input_df_ss.shape[0]):
    which_indiv = np.argwhere((test_input_df_ss.iloc[snp,:] == -1).to_numpy()).flatten()
    if (len(which_indiv) > 2):
        r_2_1 = (scipy.stats.pearsonr(list(beagle_pred.iloc[snp,which_indiv]), list(true_ss.iloc[snp,which_indiv]))[0])**2
        r_2_2 = (scipy.stats.pearsonr(list(cnn_pred_bin.iloc[snp,which_indiv]), list(true_ss.iloc[snp,which_indiv]))[0])**2
        r_2_l_1.append(r_2_1)
        r_2_l_2.append(r_2_2)
# snp_test_pred

import math
r_2_l_1_ind_new = [i for i in range(len(r_2_l_1)) if math.isnan(r_2_l_1[i]) == False]
r_2_l_2_ind_new = [i for i in range(len(r_2_l_2)) if math.isnan(r_2_l_2[i]) == False]

r_2_l_ind = set(r_2_l_1_ind_new).intersection(set(r_2_l_2_ind_new))

indices = np.array(list(r_2_l_ind))
snp_ind_wo_nan = list(np.array(range(len(r_2_l_1)))[indices.astype(int)])

r_2_l_1_new = np.array(r_2_l_1)[indices.astype(int)]
r_2_l_2_new = np.array(r_2_l_2)[indices.astype(int)]


import statistics
statistics.mean(r_2_l_1_new)
statistics.mean(r_2_l_2_new)


## Evaluate R^2 across SNPs
r_2_l_1 = []
r_2_l_2 = []


for indiv in range(test_input_df_ss.shape[1]):
    which_snp = np.argwhere((test_input_df_ss.iloc[:,indiv] == -1).to_numpy()).flatten()
    r_2_1 = (scipy.stats.pearsonr(list(beagle_pred.iloc[which_snp,indiv]), list(true_ss.iloc[which_snp,indiv]))[0])**2
    r_2_2 = (scipy.stats.pearsonr(list(cnn_pred_bin.iloc[which_snp,indiv]), list(true_ss.iloc[which_snp,indiv]))[0])**2
    r_2_l_1.append(r_2_1)
    r_2_l_2.append(r_2_2)
# snp_test_pred

import math
r_2_l_1_ind_new = [i for i in range(len(r_2_l_1)) if math.isnan(r_2_l_1[i]) == False]
r_2_l_2_ind_new = [i for i in range(len(r_2_l_2)) if math.isnan(r_2_l_2[i]) == False]

r_2_l_ind = set(r_2_l_1_ind_new).intersection(set(r_2_l_2_ind_new))

indices = np.array(list(r_2_l_ind))
snp_ind_wo_nan = list(np.array(range(1008))[indices.astype(int)])

r_2_l_1_new = np.array(r_2_l_1)[indices.astype(int)]
r_2_l_2_new = np.array(r_2_l_2)[indices.astype(int)]

import statistics
statistics.mean(r_2_l_1_new)
statistics.mean(r_2_l_2_new)

