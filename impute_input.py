from smtplib import SMTPSenderRefused
from tokenize import group
import pandas as pd
import random
import pickle
import numpy as np
import scipy.stats
import argparse
import os
import vcf
import timeit
import math
import keras


# pip3 install PyVCF

### INPUT OPTION HANDLES
os.getcwd()
os.chdir("/u/home/f/fujy2038/project-zarlab/Genotype_imputation/Impute_code_git_030223")

# from impute_function import impute_method1
from impute_function import *

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--ref', required=True, dest='reference', help='Specify the path to the training reference (reference panel file) (1000 genomes haplotype format (ending in .hap) or vcf format (ending in .vcf or .vcf.gz), or in pkl file for large python files). Individuals should be on the rows, SNPs on the columns.')
parser.add_argument('-l', '--legend', required=False, dest='legend', help='Specify the path to the legend file (required for .hap or .hap.gz files). Without legend file, will default use all positions.')
parser.add_argument('-t', '--train_input', required=True, dest='input', help='Specify the path to the training input file (masked genotypes corresponding to the reference, use -1 to mask) (1000 genomes format), or in pkl file for large python files')
parser.add_argument('-i', '--test_input', required=True, dest='test_input', help='Specify the path to the testing input file (masked genotypes that the user wants to impute) (1000 genomes format), or in pkl file for large python files')
parser.add_argument('-c', '--chr', required=False, dest='chr', help='Specify the chromosome that the user want to impute. ')
parser.add_argument('-p', '--pos', required=False, dest='pos', help='Specify the position. Format: START:END. Put "all" if you want to impute all positions in the chromosome')
parser.add_argument('--ind', required=False, dest='ind', default="all", help='Specify the path to a list of individuals that the user would like to impute. Put "all" if you want to impute all individuals in the file.')
# parser.add_argument('-s', '--skiprow', required=True, dest='skiprow', help='Specify the number of rows to skip before reading the first line')
# parser.add_argument('-n', '--nrow', required=True, dest='nrow', help='Specify the number of rows to read')
parser.add_argument('-o', '--output', required=True, dest='output', help='Specify the path to the output file (the imputed file), with filename but without suffix')
parser.add_argument('-d', '--delimiter', required=False, dest='delimiter', help='Specify the delimiter of the input files (default is any white space r"\s+")', default=r"\s+")
parser.add_argument('-g', '--group_size', required=False, dest='group_size', help='Specify the group size of the cnn local model (default is 20)', default=20)
parser.add_argument('-w', '--window_size', required=False, dest='window_size', help='Specify the window size of the cnn local model (default is 20)', default=20)
parser.add_argument('-e', '--epoch', required=False, dest='epoch', help='Specify the number of epochs of the cnn and cnn local model (default is 5)', default=5)
parser.add_argument('-b', '--batch_size', required=False, dest='batch_size', help='Specify the batch size of the cnn and cnn local model (default is 32)', default=32)
parser.add_argument('--lr', required=False, dest='lr', help='Specify the learning rate (lr) of the cnn and cnn local model (default is 1e-3)', default=1e-3)
parser.add_argument('--kr', required=False, dest='kr', help='Specify the kr of the cnn and cnn local model (default is 1e-4)', default=1e-4)
parser.add_argument('--drop_perc', required=False, dest='drop_perc', help='Specify the drop percentage of the cnn and cnn local model (default is 0.25)', default=0.25)
parser.add_argument('--model_layers', required=False, dest='model_layers', help='Specify the model layers of the cnn and cnn local model (default is 5)', default=5)
parser.add_argument('--gamma', required=False, dest='gamma', help='Specify the gamma of the sparse categorical focal loss in cnn and cnn local model (default is 5)', default=5)
parser.add_argument('--optimizer', required=False, dest='optimizer', help='Specify the optimizer of cnn and cnn local model (default is adam)', default='adam')
parser.add_argument('--hidden_layer_s', required=False, dest='hidden_layer_s', help='Specify the hidden layer size of the sparse categorical focal loss in cnn and cnn local model (default is 32)', default=32)
parser.add_argument('--to_categorical', required=False, dest='to_categorical', help='Whether use categorical encoding (0: 100, 1: 010, -1: 001), or haplotype encoding (-1:00, 0:10, 1:11, 2:01) (default is False, which means it defaults use categorical encoding). ', default=False)
parser.add_argument('--loss_func', required=False, dest='loss_func', help='Whether use focal loss (focal_loss) or cross entropy (cross_entropy).', default="focal_loss")

### Latest parameters
args = parser.parse_args(['-r', './beagle_data/processed/train_output_90_new_100.pkl', 
        '-t', './beagle_data/processed/train_input_90_new_100.pkl', 
        '-c', 'all',
        '-p', 'all',
        '-i', './beagle_data/processed/test_input_90_100.pkl', 
        '-o', './output/test_output_cnn_imputed_90',
        '-d', '\t',
        '-w', '20',
        '-g', '20',
        '-e', '5',
        '-b', '32',
        '--lr', '1e-3',
        '--kr', '1e-4',
        '--drop_perc', '0.25',
        '--model_layers', '5',
        '--gamma', '5',
        '--optimizer', 'adam',
        '--hidden_layer_s', '32',
        '--to_categorical', 'True',
        '--loss_func', 'focal_loss'
        ])

# args = parser.parse_args()

### READ THE INPUTS
ref_file = args.reference
chr_info = args.chr
pos_info = args.pos
train_input_file = args.input
test_input_file = args.test_input

num_epoch = int(args.epoch)
group_size = int(args.group_size)
window_size = int(args.window_size)
batch_size = int(args.batch_size)
lr = float(args.lr)
kr = float(args.kr)
drop_perc = float(args.drop_perc)
model_layers = int(args.model_layers)
gamma = int(args.gamma)
optimizer = args.optimizer
hidden_layer_s = int(args.hidden_layer_s)
to_categorical_bool = bool(args.to_categorical)
loss_func = args.loss_func

output_file = args.output
ind_file = args.ind
legend_file = args.legend

delim = args.delimiter

### PROCESS THE INPUTS
if pos_info == "all":
    start = 0
    end = 99999999
else:
    start = int(pos_info.split(":")[0])
    end = int(pos_info.split(":")[1])

chr = chr_info


### TEST CODE FOR READING VCF
# %%timeit
if ref_file.endswith((".vcf", ".vcf.gz")):
    ref = vcf.Reader(filename=ref_file)
    first_alleles = []
    second_alleles = []
    for record in ref.fetch(chr, start, end):
        allele1 = []
        allele2 = []
        fmt = record.FORMAT
        for sample in record.samples:
            al1 = int(sample[fmt].split("|")[0])
            al2 = int(sample[fmt].split("|")[1])
            allele1.append(al1)
            allele2.append(al2)
        first_alleles.append(allele1)
        second_alleles.append(allele2)
    samples = ref_file.samples
    first_als_df = pd.DataFrame(first_alleles)
    second_als_df = pd.DataFrame(second_alleles)
elif ref_file.endswith((".hap", ".hap.gz")):
    if legend_file is None:
        ref = pd.read_csv(ref_file, delimiter=delim, header = None,low_memory=False)
        print("Read ref!")
        test_inp = pd.read_csv(test_input_file, delimiter=delim, header = None,low_memory=False)
        print("Read test inp!")
        train_inp = pd.read_csv(train_input_file, delimiter=delim, header = None,low_memory=False)
        print("Read train inp!")
    else:
        leg = pd.read_csv(legend_file, delimiter=r"\s+", low_memory=False)
        skip_pos = []
        skip_pos.append(leg[leg['position'] < start].index.values)
        skip_pos.append(leg[leg['position'] > end].index.values)
        use_pos = leg[(leg['position'] > start) & (leg['position'] < end)].index.values
        skip_pos = [ss for pos in skip_pos for ss in pos]
        ref = pd.read_csv(ref_file, delimiter=delim, header = None, skiprows=skip_pos)
        test_inp = pd.read_csv(test_input_file, delimiter=delim, header = None, skiprows=skip_pos)
        train_inp = pd.read_csv(train_input_file, delimiter=delim, header = None, skiprows=skip_pos)
elif ref_file.endswith((".pkl")):
    if legend_file is None:
        filehandler = open(ref_file,"rb")
        ref = pickle.load(filehandler)
        filehandler.close()
        print("Read ref!")
        ref=pd.DataFrame(ref,copy=False)
        filehandler = open(test_input_file,"rb")
        test_inp = pickle.load(filehandler)
        filehandler.close()
        test_inp=pd.DataFrame(test_inp,copy=False)
        print("Read test inp!")
        filehandler = open(train_input_file,"rb")
        train_inp = pickle.load(filehandler)
        filehandler.close()
        train_inp=pd.DataFrame(train_inp,copy=False)
        print("Read train inp!")
    else:
        leg = pickle.load(legend_file)
        skip_pos = []
        skip_pos.append(leg[leg['position'] < start].index.values)
        skip_pos.append(leg[leg['position'] > end].index.values)
        use_pos = leg[(leg['position'] > start) & (leg['position'] < end)].index.values
        skip_pos = [ss for pos in skip_pos for ss in pos]
        filehandler = open(ref_file,"rb")
        ref = pickle.load(filehandler)
        filehandler.close()
        ref=pd.DataFrame(ref)
        print("Read ref!")
        filehandler = open(test_input_file,"rb")
        test_inp = pickle.load(filehandler)
        filehandler.close()
        test_inp=pd.DataFrame(test_inp)
        print("Read test inp!")
        filehandler = open(train_input_file,"rb")
        train_inp = pickle.load(filehandler)
        filehandler.close()
        train_inp=pd.DataFrame(train_inp)
        print("Read train inp!")
    ## leg.iloc[use_pos,:] ## Think about how to use legend later
else:
    raise ValueError("Reference filename not in correct format. Use '.hap', '.hap.gz', '.vcf', '.vcf.gz' or '.pkl'.")

# inp = pd.read_csv(input_file, delimiter=delim, header = None, skiprows=0)

## CNN
out_name_cnn = '_cnn_epoch%d_group%d_lr%f_kr%f_drop%f_layers%d_gamma%d_optimizer%s_hidden%d_local.txt' % (num_epoch, group_size, lr, kr, drop_perc, model_layers, gamma, optimizer, hidden_layer_s)
model_path_name_cnn = '_cnn_epoch%d_group%d_lr%f_kr%f_drop%f_layers%d_gamma%d_optimizer%s_hidden%d_local/' % (num_epoch, group_size, lr, kr, drop_perc, model_layers, gamma, optimizer, hidden_layer_s)

## FNN
# out_name_fnn = '_fnn_epoch%d_group%d_lr%d_kr%d_drop%f_layers%d_gamma%d_optimizer%s_hidden%d_local.txt' % (num_epoch, group_size, lr, kr, drop_perc, model_layers, gamma, optimizer, hidden_layer_s)
# model_path_name_fnn = '_fnn_epoch%d_group%d_lr%d_kr%d_drop%f_layers%d_gamma%d_optimizer%s_hidden%d_local/' % (num_epoch, group_size, lr, kr, drop_perc, model_layers, gamma, optimizer, hidden_layer_s)

## CNN global
# output_cnn = impute_method_cnn(train_input=train_inp, train_output=ref, test_input=test_inp, output=output_file+'_method_cnn_new.txt', model_path=output_file+'_cnn_new.h5')

## CNN local
output_cnn_group = impute_method_cnn_group_local(train_input=train_inp, \
    train_output=ref, \
    test_input=test_inp, \
    output_path=output_file+'/', \
    output=output_file+'/output'+out_name_cnn, \
    model_path=output_file+'/model'+model_path_name_cnn, \
    test_indx_path=output_file+'/test_indx.pkl', \
    group_size=group_size, \
    window_size=window_size, \
    batch_size=batch_size, \
    n_epoch=num_epoch, \
    lr=lr , \
    kr=kr , \
    drop_perc=drop_perc , \
    model_layers=model_layers , \
    gamma=gamma , \
    optimizer=optimizer , \
    hidden_layer_s=hidden_layer_s , \
    seed=20)

## FNN
# output_fnn = impute_method_fnn(train_input=train_inp, \
#     train_output=ref, \
#     test_input=test_inp, \
#     output_path=output_file+'/', \
#     output=output_file+'/output'+out_name_fnn, \
#     model_path=output_file+'/model'+model_path_name_fnn, \
#     test_indx_path=output_file+'/test_indx.pkl', \
#     group_size=group_size, \
#     window_size=window_size, \
#     batch_size=batch_size, \
#     n_epoch=num_epoch, \
#     seed=20, \
#     to_categorical=to_categorical_bool, \
#     loss_func=loss_func
#     )

## BASELINE
# output_baseline = impute_baseline(reference=ref, individual=test_inp, output=output_file+'/baseline.txt')
