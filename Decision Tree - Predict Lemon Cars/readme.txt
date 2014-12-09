Don't Get Kicked

Yogesh Rane (yrane2)

I have pre-processed the data and our program will use these pre-processed training and test csv files.

For all the classifiers:	

test.csv (original: to get the RefID attribute in the final submission file)

For DecisionTree classifiers, the following files are required in the same folder:

	training_equal_freq.csv
	test_equal_freq.csv

Data cleaning and Preprocessing
----------------------------------------------------------------
I used matlab's automated data import method to convert the given csv files into .mat files 

	test.mat 
	training.mat

Scripts (Run in Matlab)

	clean_data_categorical.m	Converts continuous variables in test.mat and training.mat into discrete variables using 
					equal frequency binning for use with decision trees

					Output is saved as training_equal_freq.csv and test_equal_freq.csv


	clean_data_categorical_interval.m	Converts continuous variables in test.mat and training.mat into discrete variables using 
						equal interval binning for use with decision trees

						Output is saved as training_equal_interval.csv and test_equal_interval.csv

	clean_data_distance.m			Converts categorical variables in test.mat and training.mat into binary variables and performs
						zscore normalization on the numeric variables

						Output is saved as zscoretraining.mat and zscoretest.mat


Classifiers
----------------------------------------------------------------
Python files
	DecisionTree.py Execute as
						python DecisionTree.py(requires training_equal_freq.csv and test_equal_freq.csv to be present in the same directory)					
					Outputs
						The program will create a submission csv file, submission_final.csv
