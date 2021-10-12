from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def roc_auc_plot(*models, X_test=None, y_test=None, title='ROC Curve'):
   
    """
    Function that accepts model(s) and test data. It will then:
    - Calculate ROC
    - Calculate AUC
    - Plot ROC curve with AUC provided in the legend

    """

    rndm_probs = [0 for _ in range(len(y_test))]
    rndm_auc = roc_auc_score(y_test, rndm_probs)
    rndm_fpr, rndm_tpr, _ = roc_curve(y_test, rndm_probs)

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.plot(rndm_fpr, rndm_tpr, linestyle='--', label='Random Chance - AUC = %.1f' % (rndm_auc))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=16, fontweight='bold')

    for i in models:
        model_name = type(i).__name__
        probs = i.predict_proba(X_test)
        probs = probs[:, 1]
        auc = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, marker='.', label=model_name + ' - AUC = %.4f' % (auc))

    ax.legend(loc='lower right', prop={'size': 15})


def classifier_train_report(*models, training_data_X = None, training_data_y = None, test_data_X = None, test_data_y = None, title = 'Reports'):
    """
    function that accepts classifier models, training data, and test data. It will then:
    - Fit the model(s) to training data
    - Make predictions using test data
    - Produce classification report for comparison
    - Produce confusion matrix for comparison
    
    """
    
    print('~'*50 + title + '~'*50)
    
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

def validation_plot(model = None, param = None, param_grid = None, X_train = None, y_train = None, title = 'Validation Curve'):
    train_scores, test_scores = validation_curve(
    model, X_train, y_train, param_name=param, param_range=param_grid,
    scoring="accuracy", cv=5)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.subplots(1, figsize=(7,7))
    plt.plot(param_grid, train_mean, label="Training score", color="black")
    plt.plot(param_grid, test_mean, label="Validation score", color="brown")

    plt.fill_between(param_grid, train_mean - train_std, train_mean + train_std, color="blue", alpha = 0.2)
    plt.fill_between(param_grid, test_mean - test_std, test_mean + test_std, color="darkblue", alpha = 0.2)
 
    plt.title(title)
    plt.xlabel("Param Range")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()