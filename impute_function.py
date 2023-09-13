import collections
from math import floor
# import numpy as np
import pandas as pd
import random
import pickle
import scipy.stats
import argparse
import os
import numpy as np
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from os.path import exists
from focal_loss import SparseCategoricalFocalLoss
from focal_loss import BinaryFocalLoss
# ref = pd.read_csv(ref_file, delimiter=delim, header = None, skiprows=skip_row, nrows=nrow)
# ind = pd.read_csv(input_file, delimiter=delim, header = None, skiprows=skip_row, nrows=nrow)


def impute_baseline(reference, individual, output):
    ref=reference.T
    ind=individual.T
    out=output
    geno_freq = []
    for snp in range(ind.shape[0]):
        snp_ss = ref.iloc[snp,:]
        cter = collections.Counter(snp_ss)
        geno_freq.append(dict(cter))
    snp_test_pred_method_1 = ind.copy()
    for indiv in range(ind.shape[1]):
        # indiv=0
    ## Generate a list of dictionaries, each dictionary correspond to 
        all_snps = list(ref)
        # counter = 0
        # for snp in all_snps.flatten():
        for snp in range(ind.shape[0]):
            # snp=which_snp.flatten()[0]
            # snp_names = snp_test_pred_method_1.iloc[snp,indiv]
            snp_ss = snp_test_pred_method_1.iloc[snp,indiv]
            possible_cases_num = [0.0,1.0]
            for x in possible_cases_num:
                if x not in geno_freq[snp]:
                    print("not in")
                    geno_freq[snp][x] = 0
            freq_l = [geno_freq[snp][x] for x in possible_cases_num]
            if sum(freq_l) > 0:
                p_l = [i / sum(freq_l) for i in freq_l]
                snp_test_pred_method_1.iloc[snp,indiv] = np.random.choice(possible_cases_num, size=None, replace=True, p=p_l)
            else:
                snp_lst = list(ref.iloc[snp,:])
                pred_snp = max(set(snp_lst), key=snp_lst.count)
                snp_test_pred_method_1.iloc[snp,indiv] = pred_snp 
            # counter += 1
    snp_test_pred_method_1.to_csv(output, index=False)
    return snp_test_pred_method_1


def convert_dosage_to_binomial(GT_input):
    GT = np.reshape(GT_input,[GT_input.shape[0],GT_input.shape[1],GT_input.shape[2]])
    ds_to_allel_presence_prob = {0:[1,0], 1:[1,1], 2:[0,1], -1: [0,0]}
    results = np.zeros([GT.shape[0],GT.shape[1],GT_input.shape[2],2])
    for old, new in ds_to_allel_presence_prob.items():
        results[GT == old] = new
    return results

def convert_binomial_to_dosage(GT_input): ## TODO: Naive version, right now only deals with haplotypes, not genotypes
    results = GT_input[:,:,:,1]
    return results

### IMPUTATION METHOD FOR CNN IN LOCAL REGIONS ###
def impute_method_cnn_group_local(train_input, train_output, test_input, output, output_path, model_path, test_indx_path, group_size=100, window_size=20, batch_size=32, n_epoch=5, seed=819, lr=1e-3, kr = 1e-4, drop_perc = 0.25, model_layers=5, gamma=5, optimizer="adam", hidden_layer_s=32):
    ## Generate training and validation individuals
    # train_input = train_input.T
    # train_output = train_output.T
    # test_input = test_input.T
    os.makedirs(model_path,exist_ok=True)
    random.seed(seed)

    train_input = np.array(train_input, copy=False)
    train_output = np.array(train_output, copy=False)

    test_input = np.array(test_input, copy=False)

    num_window = int(train_input.shape[1]/window_size)
    window_size=int(window_size)

    num_total_snps = train_input.shape[1]

    ## Subset a number of SNPs (so that it's a multiply of window size)
    num_snps_impute = floor(num_total_snps/window_size)*window_size ## Expand the number of SNPs

    ## Expand training and testing data so that it satisfies the group size (a multiply of 100 individuals)
    num_train_ind = train_input.shape[0]
    num_test_ind = test_input.shape[0]

    num_train_group = floor(num_train_ind/group_size)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if num_train_ind - num_train_group*group_size > 0:
        new_ind_sample = (num_train_group+1)*group_size
        gen_sample = new_ind_sample-num_train_ind
        sampled_ind = random.sample(range(num_train_ind), gen_sample)
        filehandler = open(output_path+"train_ind_sampled.pkl", "wb")
        pickle.dump(sampled_ind, filehandler)
        filehandler.close()
        train_input =np.concatenate([train_input, train_input[sampled_ind,:]], axis=0)
        train_output = np.concatenate([train_output, train_output[sampled_ind,:]], axis=0)

    num_test_group = floor(num_test_ind/group_size)
    if num_test_ind - num_test_group*group_size > 0:
        new_ind_sample = (num_test_group+1)*group_size
        gen_sample = new_ind_sample-num_test_ind
        sampled_ind = random.sample(range(num_test_ind), gen_sample)
        filehandler = open(output_path+"test_ind_sampled.pkl", "wb")
        pickle.dump(sampled_ind, filehandler)
        filehandler.close()
        test_input = np.concatenate([test_input, test_input[sampled_ind,:]], axis=0)

    train_input = train_input[:,0:num_snps_impute]
    train_output = train_output[:,0:num_snps_impute]
    test_input = test_input[:,0:num_snps_impute]

    test_out=np.zeros(test_input.shape)
    train_out=np.zeros(train_input.shape)

    ## Generate number of training and validation individuals
    num_train = int(num_train_ind/5*4)
    num_valid = num_train_ind-num_train
    for n in range(num_window):
        print(n)
        ## Train a CNN model in each window
        train_input_new = train_input[0:num_train, n*window_size:n*window_size+window_size]
        valid_input_new = train_input[num_train:num_train+num_valid, n*window_size:n*window_size+window_size]
        train_output_new = train_output[0:num_train, n*window_size:n*window_size+window_size]
        valid_output_new = train_output[num_train:num_train+num_valid, n*window_size:n*window_size+window_size]
        ###################
        train_group_idx = []
        valid_group_idx = []
        ###################
        train_shuff_idx = list(range(0,num_train))
        random.seed(819)
        random.shuffle(train_shuff_idx)
        ###################
        train_num_group = int(num_train/group_size)
        train_shuff_lst = [train_shuff_idx[i::train_num_group] for i in range(train_num_group)]
        ###################
        valid_shuff_idx = list(range(0,num_valid))
        random.seed(820)
        random.shuffle(valid_shuff_idx)
        ###################
        valid_num_group = int(num_valid/group_size)
        valid_shuff_lst = [valid_shuff_idx[i::valid_num_group] for i in range(valid_num_group)]
        ###################
        train_input_lst = [train_input_new[i,:] for i in train_shuff_lst]
        train_input_lst = np.array(train_input_lst)
        # train_onehot = np.array(convert_dosage_to_binomial(train_input_lst))
        train_onehot = np.array(to_categorical(train_input_lst, num_classes=3), copy=False)
        ### train_onehot shape is (3000, 100, 1000, 3)
        ###################
        train_output_lst = [train_output_new[i,:] for i in train_shuff_lst]
        train_output_lst = np.array(train_output_lst)
        # train_out_onehot = np.array(convert_dosage_to_binomial(train_output_lst))
        train_out_onehot = np.array(to_categorical(train_output_lst, num_classes=3), copy=False)
        #############
        valid_input_lst = [valid_input_new[i,:] for i in valid_shuff_lst]
        valid_input_lst = np.array(valid_input_lst)        
        # valid_onehot = np.array(convert_dosage_to_binomial(valid_input_lst))
        valid_onehot = np.array(to_categorical(valid_input_lst, num_classes=3), copy=False)
        valid_output_lst = [valid_output_new[i,:] for i in valid_shuff_lst]
        valid_output_lst = np.array(valid_output_lst)
        # valid_out_onehot = np.array(convert_dosage_to_binomial(valid_output_lst))
        valid_out_onehot = np.array(to_categorical(valid_output_lst, num_classes=3), copy=False)
        #
        lr = lr
        epochs = n_epoch
        # conv1D
        feature_size_1 = train_onehot.shape[1]
        feature_size_2 = train_onehot.shape[2]
        inChannel = train_onehot.shape[3]
        kr = kr
        drop_perc = drop_perc
        #######
        ### encoder
        ##### Version: w scaling ######
        # model = keras.Sequential()
        # model.add(layers.Conv2D(hidden_layer_s, 3, padding='same',activation='relu', input_shape=(feature_size_1, feature_size_2, inChannel), kernel_regularizer=l1(kr)))
        # model.add(layers.MaxPooling2D((2,2)))
        # model.add(layers.Dropout(drop_perc))
        # if model_layers > 2:
        #     model.add(layers.Conv2D(hidden_layer_s*2, 3, padding='same',activation='relu', kernel_regularizer=l1(kr)))
        #     if model_layers > 3:
        #         model.add(layers.MaxPooling2D((2,2)))
        #         model.add(layers.Dropout(drop_perc))
        #         if model_layers > 4:
        #             # bridge
        #             model.add(layers.Conv2D(hidden_layer_s*4, 3, padding='same',activation='relu', kernel_regularizer=l1(kr)))                  
        #             # decoder
        #         model.add(layers.Conv2D(hidden_layer_s*2, 3, padding='same',activation='relu', kernel_regularizer=l1(kr)))
        #         model.add(layers.UpSampling2D((2,2)))
        #         model.add(layers.Dropout(drop_perc))
        # else:
        #     pass
        #
        # model.add(layers.Conv2D(hidden_layer_s, 3, padding='same',activation='relu', kernel_regularizer=l1(kr))) 
        # model.add(layers.UpSampling2D(2))
        # model.add(layers.Dropout(drop_perc))
        # model.add(layers.Conv2D(inChannel, 3, padding='same', activation='softmax')) 
        #

        ##### Version: w/o scaling ######
        model = keras.Sequential()
        model.add(layers.Conv2D(hidden_layer_s, 3, padding='same',activation='relu', input_shape=(feature_size_1, feature_size_2, inChannel), kernel_regularizer=l1(kr)))
        model.add(layers.Dropout(drop_perc))
        if model_layers > 2:
            model.add(layers.Conv2D(hidden_layer_s*2, 3, padding='same',activation='relu', kernel_regularizer=l1(kr)))
            if model_layers > 3:
                model.add(layers.Dropout(drop_perc))
                if model_layers > 4:
                    # bridge
                    model.add(layers.Conv2D(hidden_layer_s*4, 3, padding='same',activation='relu', kernel_regularizer=l1(kr)))                  
                    # decoder
                model.add(layers.Conv2D(hidden_layer_s*2, 3, padding='same',activation='relu', kernel_regularizer=l1(kr)))
                model.add(layers.Dropout(drop_perc))
        else:
            pass

        model.add(layers.Conv2D(hidden_layer_s, 3, padding='same',activation='relu', kernel_regularizer=l1(kr))) 
        model.add(layers.Dropout(drop_perc))
        model.add(layers.Conv2D(inChannel, 3, padding='same', activation='softmax')) 

        # compile
        if optimizer == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == "sgd":
            optimizer = keras.optimizers.sgd(learning_rate=lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)
            print("Use default: Adam") 

        model.compile(loss=SparseCategoricalFocalLoss(gamma=gamma), 
                            optimizer=optimizer,
                            metrics=['sparse_categorical_accuracy'])
        model.summary()
        if exists(model_path+'/cnn_wo_scale_model_wsize'+ str(window_size) + 'w'+str(n)+'epoch'+str(n_epoch)+'bs'+str(batch_size)+'.h5') == False:
            model_train = model.fit(
                x=train_onehot,
                # y=train_out_onehot,
                y=train_output_lst,
                # validation_data=(valid_onehot, valid_out_onehot),
                validation_data=(valid_onehot, valid_output_lst),
                epochs=epochs,
                verbose=1,
                batch_size=batch_size            
                #     callbacks=[EarlyStopping, ModelCheckpoint]
            )
            model.save(model_path+'/cnn_wo_scale_model_wsize'+ str(window_size) + 'w'+str(n)+'epoch'+str(n_epoch)+'bs'+str(batch_size)+'.h5')  # creates a HDF5 file 'SCDA.h5'
        model = load_model(model_path+'/cnn_wo_scale_model_wsize'+ str(window_size) + 'w'+str(n)+'epoch'+str(n_epoch)+'bs'+str(batch_size)+'.h5')  # creates a HDF5 file 'SCDA.h5'
        test_input = np.array(test_input, copy = False)
        test_input_new = test_input[:,n*window_size:n*window_size+window_size]
        test_shuff_idx = list(range(0,test_input_new.shape[0]))
        test_num_group = int(test_input_new.shape[0]/group_size)
        test_shuff_lst = [test_shuff_idx[i::test_num_group] for i in range(test_num_group)]
        #
        test_input_lst = [test_input[i,n*window_size:n*window_size+window_size] for i in test_shuff_lst]
        test_input_lst = np.array(test_input_lst)
        # test_onehot = np.array(convert_dosage_to_binomial(test_input_lst))
        test_onehot = np.array(to_categorical(test_input_lst, num_classes=3), copy=False)
        ## Debugging code for generating predictions in training samples
        #         for i in range(train_onehot.shape[0]):
        #             predict_onehot = model.predict(train_onehot[i:i + 1, :, :, :])
        #             # predict_onehot = np.rint(predict_onehot)
        #             # prediction = convert_binomial_to_dosage(predict_onehot)
        #             prediction = np.argmax(predict_onehot, axis=3)
        #             prediction_reshape = prediction.reshape((prediction.shape[1], prediction.shape[2]))
        #             train_out[train_shuff_lst[i],n*window_size:n*window_size+window_size] = prediction_reshape
        # train_output[train_shuff_lst[i],n*window_size:n*window_size+window_size]
        #
        ### Generate predictions in testing samples
        for i in range(test_onehot.shape[0]):
            # print(i)
            # predict
            predict_onehot = model.predict(test_onehot[i:i + 1, :, :, :])
            predict_onehot = np.rint(predict_onehot)

            ## In binomial-dosage encoding
            prediction = convert_binomial_to_dosage(predict_onehot)

            ## In one-hot encoding
            prediction = np.argmax(predict_onehot, axis=3)

            prediction_reshape = prediction.reshape((prediction.shape[1], prediction.shape[2]))
            test_out[test_shuff_lst[i],n*window_size:n*window_size+window_size] = prediction_reshape
            # filehandler=open("/u/home/f/fujy2038/project-zarlab/Genotype_imputation/Data/1kg/Impute_code/python_files/test_input_flipped_500_20_imputed_method_cnn.pkl", "wb")
            # pickle.dump(test_out_onehot, filehandler)
            # filehandler.close()
            # filehandler=open("/u/home/f/fujy2038/project-zarlab/Genotype_imputation/Data/1kg/Impute_code/python_files/test_input_flipped_500_20_imputed_method_cnn.pkl", "rb")
            # test_out_onehot = pickle.load(filehandler)
            # filehandler.close()
        # test_out = int(test_out)
    test_out_df = pd.DataFrame(test_out.T)
    test_out_df = test_out_df.astype('int')
    test_out_ori = test_out_df.iloc[:,0:num_test_ind]
    test_out_ori.to_csv(output, index=False)
# test_out_onehot_df_new = pd.read_csv(output)
    return test_out_ori


### IMPUTATION METHOD FOR FULLY CONNECTED NETWORK ####
def impute_method_fnn(train_input, train_output, test_input, output, output_path, model_path, test_indx_path, group_size=100, window_size=20, batch_size=32, n_epoch=5, seed=819, lr=1e-3, drop_perc = 0.25, model_layers=5, gamma=5, optimizer="adam", to_categorical=False, loss_func="focal_loss"): ## Fully connected neural network
    ### TODO: Don't duplicate saving of sample individuals; Implement "to_categorical" and "loss_func".
    
    ## Generate training and validation individuals
    os.makedirs(model_path,exist_ok=True)
    random.seed(seed)
    ## Format the input and output into arrays 
    train_input = np.array(train_input, copy=False)
    train_output = np.array(train_output, copy=False)
    test_input = np.array(test_input, copy=False)
    num_window = int(train_input.shape[1]/window_size)
    window_size=int(window_size)
    num_total_snps = train_input.shape[1]
    ## Subset a number of SNPs (so that it's a multiply of window size)
    num_snps_impute = floor(num_total_snps/window_size)*window_size ## Expand the number of SNPs
    ## Expand training and testing data so that it satisfies the group size (a multiply of 100 individuals)
    num_train_ind = train_input.shape[0]
    num_test_ind = test_input.shape[0]
    ## 
    num_train_group = floor(num_train_ind/group_size)
    if num_train_ind - num_train_group*group_size > 0:
        new_ind_sample = (num_train_group+1)*group_size
        gen_sample = new_ind_sample-num_train_ind
        sampled_ind = random.sample(range(num_train_ind), gen_sample)
        filehandler = open(output_path+"_train_ind_sampled.pkl", "wb")
        pickle.dump(sampled_ind, filehandler)
        filehandler.close()
        train_input =np.concatenate([train_input, train_input[sampled_ind,:]], axis=0)
        train_output = np.concatenate([train_output, train_output[sampled_ind,:]], axis=0)
    # 
    num_test_group = floor(num_test_ind/group_size)
    if num_test_ind - num_test_group*group_size > 0:
        new_ind_sample = (num_test_group+1)*group_size
        gen_sample = new_ind_sample-num_test_ind
        sampled_ind = random.sample(range(num_test_ind), gen_sample)
        filehandler = open(output_path+"_test_ind_sampled.pkl", "wb")
        pickle.dump(sampled_ind, filehandler)
        filehandler.close()
        test_input = np.concatenate([test_input, test_input[sampled_ind,:]], axis=0)
    # 
    train_input = train_input[:,0:num_snps_impute]
    train_output = train_output[:,0:num_snps_impute]
    test_input = test_input[:,0:num_snps_impute]
    test_out=np.zeros(test_input.shape)
    train_out=np.zeros(train_input.shape)
    ## Generate number of training and validation individuals
    num_train = int(num_train_ind/5*4)
    num_valid = num_train_ind-num_train

    ## Train a CNN model in each window
    for n in range(num_window):
        print(n)
        train_input_new = train_input[0:num_train, n*window_size:n*window_size+window_size]
        valid_input_new = train_input[num_train:num_train+num_valid, n*window_size:n*window_size+window_size]
        train_output_new = train_output[0:num_train, n*window_size:n*window_size+window_size]
        valid_output_new = train_output[num_train:num_train+num_valid, n*window_size:n*window_size+window_size]
        ###################
        train_group_idx = []
        valid_group_idx = []
        ###################
        train_shuff_idx = list(range(0,num_train))
        random.seed(819)
        random.shuffle(train_shuff_idx)
        ###################
        train_num_group = int(num_train/group_size)
        train_shuff_lst = [train_shuff_idx[i::train_num_group][0:group_size] for i in range(train_num_group)]
        valid_shuff_idx = list(range(0,num_valid))
        random.seed(820)
        random.shuffle(valid_shuff_idx)
        ###################
        valid_num_group = int(num_valid/group_size)
        valid_shuff_lst = [valid_shuff_idx[i::valid_num_group][0:group_size] for i in range(valid_num_group)]
        ###################
        train_input_lst = [train_input_new[i,:] for i in train_shuff_lst]
        train_input_lst = np.array(train_input_lst)
        # train_onehot = np.array(convert_dosage_to_binomial(train_input_lst))
        # train_freq_onehot = np.array(to_categorical(train_freq_int, num_classes=3))
        train_onehot = np.array(convert_dosage_to_binomial(train_input_lst))
        # train_onehot = np.array(to_categorical(train_input_lst, num_classes=3), copy=False)
        ### train_onehot shape is (3000, 100, 1000, 3)
        ###################
        train_output_lst = [train_output_new[i,:] for i in train_shuff_lst]
        train_output_lst = np.array(train_output_lst)
        # train_out_onehot = np.array(to_categorical(train_output_lst, num_classes=3), copy=False)
        train_out_onehot = np.array(convert_dosage_to_binomial(train_output_lst))
        #
        unique_train_out, counts_train_out = np.unique(train_output_lst, return_counts=True)
        dict(zip(unique_train_out, counts_train_out))
        unique_train_in, counts_train_in = np.unique(train_input_lst, return_counts=True)
        dict(zip(unique_train_in, counts_train_in))
        #############
        valid_input_lst = [valid_input_new[i,:] for i in valid_shuff_lst]
        valid_input_lst = np.array(valid_input_lst)
        # train_freq_onehot = np.array(to_categorical(train_freq_int, num_classes=3))
        # valid_onehot = np.array(to_categorical(valid_input_lst, num_classes=3), copy=False)
        valid_onehot = np.array(convert_dosage_to_binomial(valid_input_lst))
        valid_output_lst = [valid_output_new[i,:] for i in valid_shuff_lst]
        valid_output_lst = np.array(valid_output_lst)
        ##################
        # train_freq_onehot = np.array(to_categorical(train_freq_int, num_classes=3))
        # valid_out_onehot = np.array(to_categorical(valid_output_lst, num_classes=3), copy=False)
        valid_out_onehot = np.array(convert_dosage_to_binomial(valid_output_lst))
        #####################
        ## Build model
        # missing_perc = 0.9
        # training
        # batch_size = 32
        epochs = n_epoch
        feature_size_1 = train_onehot.shape[1]
        feature_size_2 = train_onehot.shape[2]
        inChannel = train_onehot.shape[3]
        n_train = train_onehot.shape[0]
        n_valid = valid_onehot.shape[0]
        # transform y_train
        train_in_flatten = np.ndarray.reshape(train_onehot, (n_train,feature_size_1*feature_size_2*inChannel))
        valid_in_flatten = np.ndarray.reshape(valid_onehot, (n_valid,feature_size_1*feature_size_2*inChannel))
        train_out_flatten = np.ndarray.reshape(train_out_onehot, (n_train,feature_size_1*feature_size_2*inChannel))
        valid_out_flatten = np.ndarray.reshape(valid_out_onehot, (n_valid,feature_size_1*feature_size_2*inChannel))
#######
        # encoder
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=(feature_size_1*feature_size_2*inChannel,)))
        model.add(layers.Dense(feature_size_1*feature_size_2*inChannel, activation = 'relu'))
        model.add(layers.Dropout(drop_perc))
        model.add(layers.Dense(feature_size_1*feature_size_2*inChannel,activation='relu'))
        model.add(layers.Dropout(drop_perc))
        model.add(layers.Dense(feature_size_1*feature_size_2*inChannel,activation='relu'))
        model.add(layers.Dropout(drop_perc))
        model.add(layers.Dense(feature_size_1*feature_size_2*inChannel, activation='sigmoid'))
        # compile
        model.compile(loss='binary_crossentropy', 
                    #loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=True),
                            optimizer=optimizer,
                            metrics=['binary_accuracy'])
        model.summary()
        if exists(model_path+'/fnn_model_wsize'+ str(window_size) + 'w'+str(n)+'epoch'+str(n_epoch)+'bs'+str(batch_size)+'.h5') == False:
            model_train = model.fit(
                x=train_in_flatten,
                y=train_out_flatten,
                validation_data=(valid_in_flatten, valid_out_flatten),
                epochs=n_epoch,
                verbose=1,
                batch_size=batch_size,
                lr=lr
            #     callbacks=[EarlyStopping, ModelCheckpoint]
            )
            model.save(model_path+'/fnn_model_wsize'+ str(window_size) + 'w'+str(n)+'epoch'+str(n_epoch)+'bs'+str(batch_size)+'.h5')  # creates a HDF5 file 'SCDA.h5'
        model = load_model(model_path+'/fnn_model_wsize'+ str(window_size) + 'w'+str(n)+'epoch'+str(n_epoch)+'bs'+str(batch_size)+'.h5')  # creates a HDF5 file 'SCDA.h5'
        test_input = np.array(test_input, copy = False)
        test_input_new = test_input[:,n*window_size:n*window_size+window_size]
        # test_onehot = np.array(to_categorical(test_input_new, num_classes=3))
        test_shuff_idx = list(range(0,test_input_new.shape[0]))
        test_num_group = int(test_input_new.shape[0]/group_size)
        test_shuff_lst = [test_shuff_idx[i::test_num_group][0:group_size] for i in range(test_num_group)]
        # filehandler = open(test_indx_path,"wb")
        # pickle.dump(test_shuff_lst, filehandler)
        # filehandler.close()
        test_input_lst = [test_input[i,n*window_size:n*window_size+window_size] for i in test_shuff_lst]
        test_input_lst = np.array(test_input_lst)
        # train_freq_onehot = np.array(to_categorical(train_freq_int, num_classes=3))
        # test_onehot = np.array(to_categorical(test_input_lst, num_classes=3), copy=False)
        test_onehot = np.array(convert_dosage_to_binomial(test_input_lst))
        num_test = test_onehot.shape[0]
        # 
        # 
        ### Debugging code for generating predictions in training samples
        # for i in range(train_onehot.shape[0]):
        #     predict_flatten = model.predict(np.ndarray.reshape(train_onehot, (num_train,feature_size_1*feature_size_2*inChannel)))
        #     predict_flatten = np.rint(predict_flatten)
        #     predict_onehot = np.reshape(predict_flatten, (-1,feature_size_1,feature_size_2,inChannel))
        #     prediction = convert_binomial_to_dosage(predict_onehot)
        #     # prediction = np.argmax(predict_onehot, axis=3)
        #     prediction_reshape = prediction.reshape((prediction.shape[1], prediction.shape[2]))
        #     train_out[train_shuff_lst[i],n*window_size:n*window_size+window_size] = prediction_reshape
        #     # train_output[train_shuff_lst[i],n*window_size:n*window_size+window_size]
        # 
        ### train_onehot shape is (3000, 100, 1000, 3)
        # test_onehot_new = test_onehot
        # test_input=np.array(test_input)
        # masked_ind = test_input ==-1
        # for ind in range(test_onehot.shape[0]):
        #     test_onehot_new[ind][masked_ind[ind]] = test_onehot[ind][masked_ind[ind]] + train_freq_onehot[masked_ind[ind]] 
        
        ## Predict in testing samples
        predict_flatten = model.predict(np.ndarray.reshape(test_onehot, (num_test,feature_size_1*feature_size_2*inChannel)))
        predict_flatten = np.rint(predict_flatten)
        unique_predict_flatten, counts_predict_flatten = np.unique(predict_flatten, return_counts=True)
        dict(zip(unique_predict_flatten, counts_predict_flatten))        
# 
        predict_onehot = np.reshape(predict_flatten, (-1,feature_size_1,feature_size_2,inChannel))
        prediction = convert_binomial_to_dosage(predict_onehot)
        unique_prediction, counts_prediction = np.unique(prediction, return_counts=True)
        dict(zip(unique_prediction, counts_prediction))        
    # 
        # prediction = np.argmax(predict_onehot, axis=3)
        test_out[test_shuff_lst,n*window_size:n*window_size+window_size] = prediction
        # filehandler=open("/u/home/f/fujy2038/project-zarlab/Genotype_imputation/Data/1kg/Impute_code/python_files/test_input_flipped_500_20_imputed_method_cnn.pkl", "wb")
        # pickle.dump(test_out_onehot, filehandler)
        # filehandler.close()
        # filehandler=open("/u/home/f/fujy2038/project-zarlab/Genotype_imputation/Data/1kg/Impute_code/python_files/test_input_flipped_500_20_imputed_method_cnn.pkl", "rb")
        # test_out_onehot = pickle.load(filehandler)
        # filehandler.close()
        # test_out = int(test_out)
    test_out_df = pd.DataFrame(test_out.T)
    test_out_df = test_out_df.astype('int')
    test_out_ori = test_out_df.iloc[:,0:num_test_ind]
    test_out_ori.to_csv(output, index=False)
# test_out_onehot_df_new = pd.read_csv(output)
    return test_out_ori