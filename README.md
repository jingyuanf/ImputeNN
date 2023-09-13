# ImputeNN
A Convolutional-Neural-Network based model for genotype imputation

Step 1: Generate training and testing data from raw (original) Beagle example data using generate_train_test_data_beagle.py

Step 2: After generating training and testing data, call run_cnn.sh (which takes in file names and parameters and calls impute_input.py)

Step 3: impute_function.py calls functions in impute_input.py, which contain main code implementations for CNN and FNN
