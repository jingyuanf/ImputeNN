### ******** ENVIRONMENT ******** ###
from curses.ascii import NAK
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
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from os.path import exists
from collections import namedtuple

from fuc import pyvcf
from pathlib import Path

### ******** INPUT OPTION HANDLES ******** ###
os.getcwd()
os.chdir("/u/home/f/fujy2038/project-zarlab/Genotype_imputation/Impute_code_git_030223/")

parser = argparse.ArgumentParser() 
parser.add_argument('-m', '--perc_mask', required=True, dest='perc_mask', help='Specify the percentage of SNPs to mask') ## Mask a percentage of SNPs

## Use this when running in batch mode (non-interactively)
# args = parser.parse_args() 

## Use this when debugging using interactive mode
args = parser.parse_args(['-m', '20'])

perc_mask = int(args.perc_mask)
num_to_mask = int(perc_mask / 100 * 100)

### ******* READ REFERENCE DATA. GENERATE REFERENCE HAPLOTYPE DATA AND USE AS TRAINING DATA. ******* ###
## TODO: Try using genotype instead of haplotype data ##
ref_path = Path('./beagle_data/processed/ref_hap_df.pkl')
if not ref_path.is_file(): # Run only if no current reference file is generated
    ref_file = "./beagle_data/original/ref.22Jul22.46e.vcf.gz" 
    ref = vcf.Reader(open(ref_file, 'rb'))
    #
    ## This reference file has 1354 SNPs and 181 individuals
    first_alleles = []
    second_alleles = []
    chrom_l_ref = []
    pos_l_ref = []
    id_l_ref = []
    ref_alle_ref = []
    alt_alle_ref = []
    #
    ## The way that I read in the reference data is now hard coded. Need to change.
    for record in ref:
        allele1 = []
        allele2 = []
        fmt = "GT" 
        chrom_l_ref.append(record.CHROM)
        pos_l_ref.append(record.POS)
        id_l_ref.append(record.ID)
        ref_alle_ref.append(record.REF)
        alt_alle_ref.append(record.ALT)
        for sample in record.samples:
            al1 = int(sample[fmt].split("|")[0])
            al2 = int(sample[fmt].split("|")[1])
            allele1.append(al1)
            allele2.append(al2)
        first_alleles.append(allele1)
        second_alleles.append(allele2)
    #
    # samples = ref_file.samples
    first_als_df_ref = pd.DataFrame(first_alleles)
    second_als_df_ref = pd.DataFrame(second_alleles)
    ref_hap_df = pd.concat([first_als_df_ref, second_als_df_ref], axis=1)
    filehandler = open("./beagle_data/processed/ref_hap_df.pkl","wb")
    pickle.dump(ref_hap_df, filehandler)
    filehandler.close()

### ******** GENERATE MASKED TESTING DATA FROM ORIGINAL TESTING DATA  ****** ###
target_masked_path = Path('./beagle_data/processed/target_masked.22Jul22.46e_%s.vcf' % (perc_mask))

if not target_masked_path.is_file(): # Run only if no current reference file is generated
    test_file = "./beagle_data/original/target.22Jul22.46e.vcf.gz" 
    test = vcf.Reader(open(test_file, 'rb'))
    #
    test_masked_reader = vcf.Reader(open(test_file, 'rb'))
    test_masked_writer = vcf.Writer(open('./beagle_data/processed/target_masked.22Jul22.46e_%s.vcf' % (perc_mask), 'w'), test_masked_reader)
    ## IMPT CODE: write masked SNPs into the VCF file
    np.random.seed(1007)
    ct=0
    for record in test:
        ct+=1
        print(ct)
        for n in range(len(record.samples)):
            sample=record.samples[n]
            fmt="GT"
            # sample[fmt] = np.random.choice([sample[fmt], "./."], p=[1-perc_mask/100, perc_mask/100])
            samp_fmt = sample.data._fields
            new_CallData = namedtuple('CallData', samp_fmt)
            new_gt = np.random.choice([sample[fmt], "./."], p=[1-perc_mask/100, perc_mask/100])
            calldata = [new_gt] + list(record.samples[0].data[1:])
            record.samples[n].data = new_CallData(*calldata)
        test_masked_writer.write_record(record)

### ****** GENERATE HAPLOTYPE DATA FROM MASKED TARGET DATA, AND USE IT AS TESTING DATA. ****** ###
### *** FUTURE TODO: Try not use haplotype but instead use just the genotype data *** ###
test_hap_path = Path("./beagle_data/processed/test_hap_df_%s.pkl" % (perc_mask))
if not test_hap_path.is_file():
    target_masked_file = './beagle_data/processed/target_masked.22Jul22.46e_%s.vcf' % (perc_mask) 
    test = vcf.Reader(open(target_masked_file, 'r'))
    #
    first_alleles = []
    second_alleles = []
    chrom_l_test = []
    pos_l_test = []
    id_l_test = []
    ref_alle_test = []
    alt_alle_test = []
    #
    ct=0
    for record in test:
        ct+=1
        print(ct)
        allele1 = []
        allele2 = []
        fmt = "GT" 
        chrom_l_test.append(record.CHROM)
        pos_l_test.append(record.POS)
        id_l_test.append(record.POS)
        ref_alle_test.append(record.REF)
        alt_alle_test.append(record.ALT) ## Hard coded here. May need to change. Also, not sure what do GT, DS and GL mean exactly. 
        for sample in record.samples:
            if sample[fmt].split("/")[0] == '0' or sample[fmt].split("/")[0] == '1':
                al1 = int(sample[fmt].split("/")[0])
                al2 = int(sample[fmt].split("/")[1])
            else:
                al1=-1
                al2=-1
            allele1.append(al1)
            allele2.append(al2)
        first_alleles.append(allele1)
        second_alleles.append(allele2)
    #
    first_als_df_test = pd.DataFrame(first_alleles)
    second_als_df_test = pd.DataFrame(second_alleles)
    test_hap_df = pd.concat([first_als_df_test, second_als_df_test], axis=1)
    #
    filehandler = open("./beagle_data/processed/test_hap_df_%s.pkl" % (perc_mask),"wb")
    pickle.dump(test_hap_df, filehandler)
    filehandler.close()

## *** LOAD TEST HAPLOTYPE DATA *** ##
filehandler = open("./beagle_data/processed/test_hap_df_%s.pkl" % (perc_mask),"rb")
test_hap_df = pickle.load(filehandler)
filehandler.close()

### ****** GENERATE TEST TRUTH FILE FOR EVALUATION ****** ####
test_truth_path = Path("./beagle_data/processed/test_eval_hap_df.pkl")
if not test_truth_path.is_file():
    test_eval_file = "./beagle_data/original/target.22Jul22.46e.vcf.gz" 
    eval = vcf.Reader(open(test_eval_file, 'rb'))
    #
    first_alleles = []
    second_alleles = []
    chrom_l_eval = []
    pos_l_eval = []
    id_l_eval = []
    ref_alle_eval = []
    alt_alle_eval = []
    for record in eval:
        allele1 = []
        allele2 = []
        fmt = "GT" 
        chrom_l_eval.append(record.CHROM)
        pos_l_eval.append(record.POS)
        id_l_eval.append(record.POS)
        ref_alle_eval.append(record.REF)
        alt_alle_eval.append(record.ALT) ## Hard coded here. May need to change. Also, not sure what do GT, DS and GL mean exactly. 
        for sample in record.samples:
            al1 = int(sample[fmt].split("/")[0])
            al2 = int(sample[fmt].split("/")[1])
            allele1.append(al1)
            allele2.append(al2)
        first_alleles.append(allele1)
        second_alleles.append(allele2)
            
    first_als_df_eval = pd.DataFrame(first_alleles)
    second_als_df_eval = pd.DataFrame(second_alleles)
    eval_hap_df = pd.concat([first_als_df_eval, second_als_df_eval], axis=1)
    ## This output file has 1356 SNPs and 10 individuals
    #
    filehandler = open("./beagle_data/processed/test_eval_hap_df.pkl","wb")
    pickle.dump(eval_hap_df, filehandler)
    filehandler.close()

### ****** PROCESS BEAGLE OUTPUT FILE IN HAPLOTYPES GENERATED WITHOUT USING REFERENCE DATA (RUN IT AFTER RUNNING BEAGLE) ****** ###
beagle_out_gt_path = Path("./beagle_data/processed/beagle_out_hap_df_gt_%s.pkl")
if not beagle_out_gt_path.is_file():
    out_file = "./beagle_data/beagle_run/out_masked_%s.gt.vcf.gz" % (perc_mask)
    out = vcf.Reader(open(out_file, 'rb'))
    #
    first_alleles = []
    second_alleles = []
    chrom_l_out = []
    pos_l_out = []
    id_l_out = []
    ref_alle_out = []
    alt_alle_out = []
    for record in out:
        allele1 = []
        allele2 = []
        fmt = "GT" 
        chrom_l_out.append(record.CHROM)
        pos_l_out.append(record.POS)
        id_l_out.append(record.POS)
        ref_alle_out.append(record.REF)
        alt_alle_out.append(record.ALT) ## Hard coded here. May need to change. Also, not sure what do GT, DS and GL mean exactly. 
        for sample in record.samples:
            al1 = int(sample[fmt].split("|")[0])
            al2 = int(sample[fmt].split("|")[1])
            allele1.append(al1)
            allele2.append(al2)
        first_alleles.append(allele1)
        second_alleles.append(allele2)
        #
        # samples = ref_file.samples
        first_als_df_out = pd.DataFrame(first_alleles)
        second_als_df_out = pd.DataFrame(second_alleles)
        out_hap_df = pd.concat([first_als_df_out, second_als_df_out], axis=1)
        ## This output file has 1356 SNPs and 10 individuals
        #
        filehandler = open("./beagle_data/processed/beagle_out_hap_df_gt_%s.pkl" % (perc_mask),"wb")
        pickle.dump(out_hap_df, filehandler)
        filehandler.close()

### ****** PROCESS BEAGLE OUTPUT FILE IN HAPLOTYPES GENERATED USING REFERENCE DATA (RUN IT AFTER RUNNING BEAGLE) ****** ###
beagle_out_ref_path = Path("./beagle_data/processed/beagle_out_hap_df_ref_%s.pkl" % (perc_mask))
if not beagle_out_ref_path.is_file():
    out_file = "./beagle_data/beagle_run/out_masked_%s.ref.vcf.gz" % (perc_mask) 
    out = vcf.Reader(open(out_file, 'rb'))
    #
    first_alleles = []
    second_alleles = []
    chrom_l_out = []
    pos_l_out = []
    id_l_out = []
    ref_alle_out = []
    alt_alle_out = []
    for record in out:
        allele1 = []
        allele2 = []
        fmt = "GT" 
        chrom_l_out.append(record.CHROM)
        pos_l_out.append(record.POS)
        id_l_out.append(record.POS)
        ref_alle_out.append(record.REF)
        alt_alle_out.append(record.ALT) ## Hard coded here. May need to change. Also, not sure what do GT, DS and GL mean exactly. 
        for sample in record.samples:
            al1 = int(sample[fmt].split("|")[0])
            al2 = int(sample[fmt].split("|")[1])
            allele1.append(al1)
            allele2.append(al2)
        first_alleles.append(allele1)
        second_alleles.append(allele2)
    #
    # samples = ref_file.samples
    first_als_df_out = pd.DataFrame(first_alleles)
    second_als_df_out = pd.DataFrame(second_alleles)
    out_hap_df = pd.concat([first_als_df_out, second_als_df_out], axis=1)
    ## This output file has 1356 SNPs and 10 individuals
    filehandler = open("./beagle_data/processed/beagle_out_hap_df_ref_%s.pkl" % (perc_mask),"wb")
    pickle.dump(out_hap_df, filehandler)
    filehandler.close()

### *********** LOAD PREVIOUSLY GENERATED DATA *********** ###

filehandler = open("./beagle_data/processed/ref_hap_df.pkl","rb")
ref_hap_df = pickle.load(filehandler)
filehandler.close()

filehandler = open("./beagle_data/processed/beagle_out_hap_df_ref_%s.pkl" % (perc_mask),"rb") ## Use the beagle file generated using reference panel
out_hap_df = pickle.load(filehandler)
filehandler.close()

filehandler = open("./beagle_data/processed/test_hap_df_%s.pkl" % (perc_mask),"rb")
test_hap_df = pickle.load(filehandler)
filehandler.close()

### *********** GENERATE TRAIN INPUT AND OUTPUT DATA *********** ###

train_hap_df = ref_hap_df

num_to_mask = int(perc_mask / 100 * ref_hap_df.shape[0])

train_input = []
train_output = []
random.seed(0)
for ind in range(train_hap_df.shape[1]):
    train_ind = train_hap_df.iloc[:,ind]
    for t in range(1000):
        mask_ind = random.sample(range(len(train_ind)), num_to_mask)
        train_mask = train_ind.copy()
        train_mask[mask_ind] = -1
        train_input.append(train_mask)
        train_output.append(train_ind)

filehandler = open("./beagle_data/processed/train_input_%s_new_100.pkl" % (perc_mask),"wb")
pickle.dump(train_input, filehandler)
filehandler.close()

filehandler = open("./beagle_data/processed/train_output_%s_new_100.pkl" % (perc_mask),"wb")
pickle.dump(train_output, filehandler)
filehandler.close()

filehandler = open("./beagle_data/processed/train_input_%s_new_100.pkl" % (perc_mask),"rb")
train_input = pickle.load(filehandler)
filehandler.close()

filehandler = open("./beagle_data/processed/train_output_%s_new_100.pkl" % (perc_mask),"rb")
train_output = pickle.load(filehandler)
filehandler.close()


### *********** GENERATE TEST INPUT AND OUTPUT DATA *********** ###
test_input = test_hap_df
test_output = out_hap_df

test_input_df = pd.DataFrame(test_input).T
test_output_df = pd.DataFrame(test_output).T

filehandler = open("./beagle_data/processed/test_input_%s_100.pkl" % (perc_mask),"wb")
pickle.dump(test_input_df, filehandler)
filehandler.close()

filehandler = open("./beagle_data/processed/test_output_%s_100.pkl" % (perc_mask),"wb")
pickle.dump(test_output_df, filehandler)
filehandler.close()

# filehandler = open("./beagle_data/processed/test_input_%s_100.pkl" % (perc_mask),"rb")
# test_input_df = pickle.load(filehandler)
# filehandler.close()

# filehandler = open("./beagle_data/processed/test_output_%s_100.pkl" % (perc_mask),"rb")
# test_output_df = pickle.load(filehandler)
# filehandler.close()

test_input_df.to_csv(path_or_buf="./beagle_data/processed/test_input_%s.hap" % (perc_mask), sep="\t")
test_output_df.to_csv(path_or_buf="./beagle_data/processed/test_output_%s.hap" % (perc_mask), sep="\t")

