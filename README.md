# ImputeNN
A Convolutional-Neural-Network based model for genotype imputation

Step 1: Generate training and testing data from raw (original) example data from Beagle paper using generate_train_test_data_beagle.py (modify the code if you want to train and test on other sets of data)

Step 2: After generating training and testing data, call run_cnn.sh (which takes in file names and parameters and calls impute_input.py)

Step 3: impute_function.py calls functions in impute_input.py, which contain main code implementations for CNN and FNN
