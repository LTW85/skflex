# skflex

![GitHub](https://img.shields.io/github/license/ltw85/skflex) ![PyPI](https://img.shields.io/pypi/v/skflex) [![Build Status](https://scrutinizer-ci.com/g/LTW85/skflex/badges/build.png?b=main)](https://scrutinizer-ci.com/g/LTW85/skflex/build-status/main) [![Tweet](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FLTW85%2Fskflex)](https://twitter.com/intent/tweet?text=https%3A%2F%2Fgithub.com%2FLTW85%2Fskflex)

## *FLEXIBLE FUNCTIONS* ----- *FAST PROCESSING AND EVALUATION*

skflex provides a suite of utility functions for use with the sklearn library. The module primarily focuses on producing typical plots and metrics for evaluating machine learning models. It has been designed with flexability and customisation in mind to speed up workflows, and enhance comparative evaluation.

## Installation and Import
```
pip install skflex

import skflex.skflex as skf
```
_____________________________________________________________________________________________________________________________________________________________________

# roc_auc_plot
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

#### Example:

```
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

model_1 = GaussianNB()
model_2 = LogisticRegression()

model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)

skf.roc_auc_plot(model_1, model_2, X_test = X_test, y_test = y_test, 
                title = 'Example ROC plot')
```

![roc plot](example_roc_plot.PNG)

_________________________________________________________________________________________________________________________________________________________________________________________________________________________

# classifier_train_report
Accepts classifier models, training data, and test data. It will then:
- Fit the model(s) to training data
- Make predictions using test data
- Produce classification report for comparison
- Produce confusion matrix for comparison
- Provide an ordered summary (ranked best to worst score) using given evaluation metric

#### Parameters:
- models - model objects to be trained and evaluated
- X_train - training feature set
- y_train - training label set
- X_test - test feature set
- y_test - test label set
- scoring - summary evaluation metric. Common classifier evaluation metrics including accuracy, f1, precision, and recall are supported. Refer to [sklean scoring documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) for more information. Scoring methodologies should be passed as strings, for example, precision would be passed as `scoring = 'precision'` 
- title - title for output report 

#### Default:

models, X_train = None, y_train = None, X_test = None, y_test = None, scoring = 'accuracy', title = 'Reports'

#### Example:

```
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

model_1 = GaussianNB()
model_2 = LogisticRegression()

skf.classifier_train_report(model_1, model_2, X_train = X_train, y_train = y_train, 
                            X_test = X_test, 
                            y_test = y_test, 
                            scoring = 'accuracy', 
                            title = 'Example Report')
```

![classification report](example_classification_report1.PNG)

_________________________________________________________________________________________________________________________________________________________________________________________________________________________

# validation_plot
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

#### Example:

```
from sklearn.tree import DecisionTreeClassifier

model_1 = DecisionTreeClassifier()
params = [5, 10, 15, 20, 30, 40, 50]

skf.validation_plot(model = model_1, param = 'max_depth', param_grid = params, 
                    X_train = X_train, 
                    y_train = y_train, 
                    title = 'Example Validation Curve')
```

![validation plot](example_validation_curve.PNG)

____________________________________________________________________________________________________________________________________________________________________________________________________________________________

# train_val_test
Accepts a Pandas dataframe and will return a training, validation, and test set. Operates in a similar fashion to the sklearn train_test_split function by defining a percentage split for the training and validation sets (example 0.6 = 60%). The remainder is allocated to the test set.

#### Parameters:
- data - dataframe to be split into a training, validation, and test set
- class_labels - column in the dataframe containing class labels
- train - percentage of data to be allocated to the training set
- val - percentage of data to be allocated to the validation set
- shuffle - if true, will shuffle the rows in the dataframe before splitting
- random_state - if shuffle is ture, will set a random seed so that ordering produced by shuffle can be reproduced

#### Default:

data = None, class_labels = None, train = 0.6, val = 0.2, shuffle = True, random_state = None

Returns: X_train, y_train, X_val, y_val, X_test, y_test

#### Example:

```
X_train, y_train, X_val, y_val, X_test, y_test = skf.train_val_test(data = my_data, 
                                                                    class_labels = 'labels', 
                                                                    train = 0.6, 
                                                                    val = 0.2)
```

______________________________________________________________________________________________________________________________________________________________________________________________________________________________

# pca_scree_plot
Accepts data (array/dataframe), and number of principal components to be analysed. It will produce a scree plot of the cumulative variance explained.

#### Parameters:
- data - dataset to be analysed
- n_components - number of principal components to be analysed 
- width - width of plot
- height - height of plot
- legend_size - size of plot legend
- scale_data - normalises data before analysis and plotting. If the data being passed has not yet been normalised, this parameter should be set as `scale_data = True`
- title - plot title

#### Default:

data = None, n_components = None, width = 16, height = 10, legend_size = 12, scale_data = False, title = 'PCA Scree Plot'

### Example:

```
from sklearn.preprocessing import scale

n_data = scale(my_data)

skf.pca_scree_plot(data = n_data, n_components = 70, title = 'Example PCA Scree Plot')
```
![scree plot](example_scree_plot.PNG)

_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## Requirements
- Sklearn >= 0.24.1
- Matplotlib >= 3.3.4
- Pandas >= 1.2.4
- Numpy >= 1.20.1
