from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import validation_curve
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import numpy as np
import pandas as pd

def roc_auc_plot(*models, X_test = None, y_test = None, width = 14, height = 12, legend_size = 14, title = 'ROC Curve'):
   
    """
    Function that accepts fitted model(s) and test data. It will then:
    - Calculate ROC
    - Calculate AUC
    - Plot ROC curve with AUC provided in the legend

    """

    rndm_probs = [0 for _ in range(len(y_test))]
    rndm_auc = roc_auc_score(y_test, rndm_probs)
    rndm_fpr, rndm_tpr, _ = roc_curve(y_test, rndm_probs)

    plt.subplots(1, figsize=(width, height))
    plt.plot(rndm_fpr, rndm_tpr, linestyle='--', label='Random Chance - AUC = %.1f' % (rndm_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=16, fontweight='bold')
   
    
    for i in models:
        model_name = type(i).__name__
        
        if hasattr( i, 'predict_proba' ) and callable( i.predict_proba ):
            probs = i.predict_proba(X_test)
            probs = probs[: , 1]
        else:
            probs = i.decision_function(X_test)
            
        auc = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, marker = '.', label= model_name + ' - AUC = %.4f' % (auc))
        
    plt.legend(loc = 'lower right', prop={'size': legend_size}) 
    plt.tight_layout() 
    plt.show()

def classifier_train_report(*models, training_data_X = None, training_data_y = None, test_data_X = None, test_data_y = None, scoring = accuracy_score, title = 'Reports'):
    
    """
    function that accepts classifier models, training data, and test data. It will then:
    - Fit the model(s) to training data
    - Make predictions using test data
    - Produce classification report for comparison
    - Produce confusion matrix for comparison
    - Provide ordered summary using given evaluation metric
    
    """
    
    print('~'*50 + title + '~'*50)
    
    m_scores = []
    for i in models:
        model_name = type(i).__name__
        i.fit(training_data_X, training_data_y)
        y_pred = i.predict(test_data_X)
        print()
        print('-'*20 + model_name + ' ' + 'Classification Report' + '-'*20)
        print(classification_report(y_pred, test_data_y)) 
        print()
        print('-'*20 + model_name + ' ' + 'Confusion Matrix' + '-'*20)
        print(confusion_matrix(y_pred, test_data_y))
        print()
        print('*'*100) 
        print()
        score = scoring(test_data_y, y_pred)
        m_scores.append({'model': model_name, 'score': score})
        
    sorted_scores = sorted(m_scores, key=lambda s: s['score'], reverse = True)
    
    print('Summary:') 
    for s in sorted_scores:
        print('model:' + ' ' + str(s['model']) + ' ' + '--- ' + 'score:' + ' ' + str(s['score'].round(3)))   

def validation_plot(model = None, param = None, param_grid = None, X_train = None, y_train = None, cv = 5, scoring = 'accuracy', width = 9, height = 9, 
                    title = 'Validation Curve'):

    """ 
    Function that accepts a model, a related hyper-parameter, a list of hyper-parameter values, training and test data, number of cross-validation folds, scoreing methodology, as well as a plot title.
    It will produce a plot of the validation curve for the training and test data using the mean scores and standard deviations obtained through the cross-validation process.  
    
    """
    train_scores, test_scores = validation_curve(
    model, X_train, y_train, param_name=param, param_range=param_grid,
    scoring=scoring, cv=cv)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.subplots(1, figsize = (width, height))
    plt.plot(param_grid, train_mean, label = 'Training score', color = 'black')
    plt.plot(param_grid, test_mean, label = 'Validation score', color = 'brown')

    plt.fill_between(param_grid, train_mean - train_std, train_mean + train_std, color = 'blue', alpha = 0.2)
    plt.fill_between(param_grid, test_mean - test_std, test_mean + test_std, color = 'darkblue', alpha = 0.2)
 
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Param Range')
    plt.ylabel('Accuracy Score')
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()

def train_val_test(data = None, class_labels = None, train = 0.6, val = 0.2, shuffle = True, random_state = None):
    
    """
    Function that accepts a Pandas dataframe and will return a training, validation, and test set. 
    """
    
    if shuffle:
        data = data.sample(frac = 1, random_state = random_state)
    else:
        pass
    
    X = data.drop(class_labels, axis = 1)
    y = pd.DataFrame(data[class_labels])
    
    split = train + val
    X_train, X_val, X_test = np.split(X, [int(train*len(X)), int(split*len(X))])
    y_train, y_val, y_test = np.split(y, [int(train*len(y)), int(split*len(y))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def pca_scree_plot(data = None, n_components = None, width = 16, height = 10, legend_size = 12, title = 'PCA Scree Plot'):

    """
    Function that accepts data, and number of principal components to be analysed. It will produce a scree plot of the cumulative variance explained.  
    """
    
    pca = PCA(n_components = n_components)
    pca_model = pca.fit(data)
    var_exp = pca_model.explained_variance_ratio_.cumsum().round(4)*100
    
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Components', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.axhspan(90, 100, alpha = 0.3, color = '#FF8C78', label = '90% - 95%')
    ax.axhspan(95, 100, alpha = 0.5, color = '#FF8C78', label = '95% - 99%')
    ax.axhspan(99, 100, alpha = 0.7, color = '#FF8C78', label = '99% - 100%')
    ax.legend(loc = 4, prop={'size': legend_size})
    ax.plot(var_exp, marker = '.', markersize = 10);
    plt.show()    