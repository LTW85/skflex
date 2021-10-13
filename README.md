# sklearn_utils

Utility functions for use with the sklearn library. Primarily focuses on producing typical plots and metrics for evaluating machine learning models. Many of the functions accept multiple models to speed up workflows and enhance comparative evaluation.

## Functions 
Functions currently included along with descriptions and default parameter settings.

### roc_auc_plot
Accepts fitted model(s) and test data. It will then:
- Calculate ROC
- Calculate AUC
- Plot ROC curve with AUC provided in the legend

#### Parameters:
- models - fitted model objects. NOTE: Only models with a 'predict_proba' or 'decision_function' method are supported.
- X_test - test feature set
- y_test - test labels set
- title - title for ROC curve plot
- width - plot width
- height - plot height
- legend_size - size of plot legend

#### Default:

models, X_test = None, y_test = None, width = 14, height = 12, legend_size = 14, title='ROC Curve'

### classifier_train_report
Accepts classifier models, training data, and test data. It will then:
- Fit the model(s) to training data
- Make predictions using test data
- Produce classification report for comparison
- Produce confusion matrix for comparison

#### Parameters:
- models - model objects to be trained and evaluated
- training_data_X - training feature set
- training_data_y - training label set
- test_data_X - test feature set
- test_data_y - test label set
- title - title for output report 

#### Default:

models, training_data_X = None, training_data_y = None, test_data_X = None, test_data_y = None, title = 'Reports'

### validation_plot
Accepts a model, a related hyper-parameter, a list of hyper-parameter values, training and test data, number of cross-validation folds, scoring methodology, as well as a plot title.
It will produce a plot of the validation curve for the training and test data using the mean scores and standard deviations obtained through the cross-validation process. 

#### Parameters:
- model - model object 
- param - hyperparameter to be used to produce the validation curve 
- param_grid - hyperparameter values to be tested
- X_train - training feature set
- y_train - training label set
- cv - number of cross-validation folds
- scoring - scoring methodology used during cross-validation process
- title - title for validation plot
- width - plot width
- height - plot height

#### Default:

model = None, param = None, param_grid = None, X_train = None, y_train = None, cv = 5, scoring = 'accuracy', width = 9, height = 9, title = 'Validation Curve'

### train_val_test
Accepts a Pandas dataframe and will return a training, validation, and test set. Operates in a similar fashion to the sklearn train_test_split function by defining a percentage split for the training and validation sets (example 0.6 = 60%). The remainder is allocated to the test set.

#### Parameters:
- data - dataframe to be split into a training, validation, and test set
- class_labels - column in the dataframe containing class labels
- train - percentage of data to be allocated to the training set
- val - percentage of data to be allocated to the validation set
- shuffle - if true, will shuffle the rows in the dataframe before splitting

#### Default:

data = None, class_labels = None, train = 0.6, val = 0.2, shuffle = True

Returns: X_train, y_train, X_val, y_val, X_test, y_test

### pca_scree_plot
Accepts data (array/dataframe), and number of principal components to be analysed. It will produce a scree plot of the cumulative variance explained.

#### Parameters:
- data - dataset to be analysed
- n_components - number of principal components to be analysed 
- width - width of plot
- height - height of plot
- legend_size - size of plot legend
- title - plot title

#### Default:

data = None, n_components = None, width = 16, height = 10, legend_size = 12, title = 'PCA Scree Plot'

## Dependencies
- Sklearn
- Matplotlib
- Pandas
- Numpy
