# Problem's brief description
Dataset contains telematics data during trips (bookingID). 
Each trip is assigned with label 1 or 0 to indicate dangerous driving and could contain thousands of telematics data points. 

# Data preprocessing
- Data cleansing for training
	- Delete trips with multiple (different) labels
	- Delete data points that have speed below zero

- Assuming the trip can be said to be dangerous if one of these conditions occurs:
	+ High speed experienced by the driver
	+ A drastic change in speed, accelerometer or gyroscope after 1 or 2 seconds later. This might indicate a sudden acceleration, sudden brake or turn suddenly.
		
- Feature engineering	
	- Sort the dataset by bookingID and second 
	- Calculate acceleration and gyroscope resultan (x, y, z)
	- Calculate all the changes (Speed, Acceleration resultan, Bearing, etc) after 1 or 2 seconds later in each rows. 
		If the difference in seconds is large (more than 2 seconds), then ignore (set the value to zero). 
		Special case for Bearing_changes, its angle change should be no more than 180.	
	- Calculate the aggregation (Min, Max, Mean, Standard Deviation, Range) based on bookingID and all columns.
# Models
Using Voting Ensemble as a meta-ensemble model consisting of Random Forest (Scikit-Learn) and Gradient Boosting (XGBoost and LightGBM).

# Hyperparameters optimization
Using the hyperopt library (http://hyperopt.github.io/hyperopt/) to optimize Random Forest, XGBoost, and LightGBM Hyperparameters.

# Project structure
- data 
	+ raw : storing the raw dataset
	+ interim : storing the intermediate result 
	+ processed : the place for training/predicting-ready dataset
- images
	+ 10-Fold Cross Validation : Some ROC-AUC chart to show the performance of the model 
- models : the binary file of the model by pickle
- output : storing the prediction values
- data_preprocessor.py : data preprocessing, feature engineering
- prepare_training_data.py : prepare the training-ready dataset for training the model
- hyperparameters_opt.py : optimize hyperparameters of models
- cross_validation.py : using 10-fold cross validation to test the reliability of the model
- model_configuration.py : configure the models with each optimized hyperparameters
- training.py : training the model
- predict.py : predict the output of dataset that is in folder 'data/raw/features/' and store the output in folder 'output/'
	
# ROC-AUC with 10-fold Cross Validation
![alt text](https://raw.githubusercontent.com/nawa410/safety-challenge/master/images/10-Fold%20Cross%20Validation/ensemble.png)

# How to predict using the model trained
- Prerequisite: Python 3, Pandas, Numpy, and Scikit-Learn.
- Put the datasets in the 'data/raw/features/' folder
- run 'python predict.py' 
- The predicted values (sorted by bookingID) can be accessed in the 'output/' folder 
