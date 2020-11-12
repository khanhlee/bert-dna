#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data_dir = 'gdrive/My Drive/radiomics/nsclc/pyradiomics/'
data_dir = 'dataset/'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# In[2]:


df_train = pd.read_csv(os.path.join(data_dir, '6mA.bert.cv.csv'), header=None)
df_test = pd.read_csv(os.path.join(data_dir, '6mA.bert.ind.csv'), header=None)


# In[3]:


X_trn_bert = df_train.iloc[:,1:]
y_trn_bert = df_train[0]
X_tst_bert = df_test.iloc[:,1:]
y_tst_bert = df_test[0]


# In[4]:


def load_data(trn_file, tst_file):
    df_trn = pd.read_csv(os.path.join(data_dir, trn_file))
    df_tst = pd.read_csv(os.path.join(data_dir, tst_file))
    
    X_trn = df_trn.drop('label', axis=1)
    y_trn = df_trn['label']
    X_tst = df_tst.drop('label', axis=1)
    y_tst = df_tst['label']
    
    return X_trn, y_trn, X_tst, y_tst


# In[5]:


X_trn_kmer, y_trn_kmer, X_tst_kmer, y_tst_kmer = load_data('6mA.kmer.cv.csv', '6mA.kmer.ind.csv')
X_trn_psednc, y_trn_psednc, X_tst_psednc, y_tst_psednc = load_data('6mA.psednc.cv.csv', '6mA.psednc.ind.csv')
X_trn_pseknc, y_trn_pseknc, X_tst_pseknc, y_tst_pseknc = load_data('6mA.pseknc.cv.csv', '6mA.pseknc.ind.csv')
X_trn_DAC, y_trn_DAC, X_tst_DAC, y_tst_DAC = load_data('6mA.DAC.cv.csv', '6mA.DAC.ind.csv')
X_trn_DACC, y_trn_DACC, X_tst_DACC, y_tst_DACC = load_data('6mA.DACC.cv.csv', '6mA.DACC.ind.csv')
X_trn_DCC, y_trn_DCC, X_tst_DCC, y_tst_DCC = load_data('6mA.DCC.cv.csv', '6mA.DCC.ind.csv')
X_trn_TAC, y_trn_TAC, X_tst_TAC, y_tst_TAC = load_data('6mA.TAC.cv.csv', '6mA.TAC.ind.csv')
X_trn_TACC, y_trn_TACC, X_tst_TACC, y_tst_TACC = load_data('6mA.TACC.cv.csv', '6mA.TACC.ind.csv')
X_trn_TCC, y_trn_TCC, X_tst_TCC, y_tst_TCC = load_data('6mA.TCC.cv.csv', '6mA.TCC.ind.csv')


# In[6]:


y_tst_bert.shape


# In[7]:


feat_dict = {'kmer':X_trn_kmer, 'DAC':X_trn_DAC, 'DACC':X_trn_DACC, 'DCC':X_trn_DCC,
            'TAC':X_trn_TAC, 'TACC':X_trn_TACC, 'TCC':X_trn_TCC, 'PseDNC':X_trn_psednc,
            'PseKNC':X_trn_pseknc, 'BERT':X_trn_bert}


# In[8]:


for key, value in feat_dict.items():
    print(key)


# ## Baseline model comparison

# In[6]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

kfold = StratifiedKFold(n_splits=5, shuffle=True)
from sklearn import metrics


# In[93]:


for key, value in feat_dict.items():
    TP = FP = TN = FN = 0
    acc_cv_scores = []
    auc_cv_scores = []
    for train, test in kfold.split(value, y_trn):
        svm_model = RandomForestClassifier(n_estimators=500) 
        ## evaluate the model
        svm_model.fit(value.iloc[train], y_trn.iloc[train])
        # evaluate the model
        true_labels = np.asarray(y_trn.iloc[test])
        predictions = svm_model.predict(value.iloc[test])
        acc_cv_scores.append(accuracy_score(true_labels, predictions))
        # print(confusion_matrix(true_labels, predictions))
        newTN, newFP, newFN, newTP = confusion_matrix(true_labels,predictions).ravel()
        TP += newTP
        FN += newFN
        FP += newFP
        TN += newTN
        pred_prob = svm_model.predict_proba(value.iloc[test])
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_prob[:,1], pos_label=1)
        auc_cv_scores.append(metrics.auc(fpr, tpr))
        # print('AUC = ', metrics.auc(fpr, tpr))
        # print('AUC = ', round(metrics.roc_auc_score(true_labels, predictions)*100,2))

    print('\nFeature: ', key)
    print('Accuracy = ', np.mean(acc_cv_scores))
    print('TP = %s, FP = %s, TN = %s, FN = %s' % (TP, FP, TN, FN))
    print('AUC = ', np.mean(auc_cv_scores))


# In[ ]:


for key, value in feat_dict.items():
    TP = FP = TN = FN = 0
    acc_cv_scores = []
    auc_cv_scores = []
    for train, test in kfold.split(value, y_trn):
        svm_model = RandomForestClassifier(n_estimators=500) 
        ## evaluate the model
        svm_model.fit(value.iloc[train], y_trn.iloc[train])
        # evaluate the model
        true_labels = np.asarray(y_trn.iloc[test])
        predictions = svm_model.predict(value.iloc[test])
        acc_cv_scores.append(accuracy_score(true_labels, predictions))
        # print(confusion_matrix(true_labels, predictions))
        newTN, newFP, newFN, newTP = confusion_matrix(true_labels,predictions).ravel()
        TP += newTP
        FN += newFN
        FP += newFP
        TN += newTN
        pred_prob = svm_model.predict_proba(value.iloc[test])
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_prob[:,1], pos_label=1)
        auc_cv_scores.append(metrics.auc(fpr, tpr))
        # print('AUC = ', metrics.auc(fpr, tpr))
        # print('AUC = ', round(metrics.roc_auc_score(true_labels, predictions)*100,2))

    print('\nFeature: ', key)
    print('Accuracy = ', np.mean(acc_cv_scores))
    print('TP = %s, FP = %s, TN = %s, FN = %s' % (TP, FP, TN, FN))
    print('AUC = ', np.mean(auc_cv_scores))


# In[82]:


from sklearn.model_selection import cross_val_score

# Comparison of the performance results among different baseline models
lr_model = LogisticRegression(max_iter=1000)
lr_result = cross_val_score(lr_model, X_trn_pseknc, y_trn_pseknc, cv=kfold)
print('Logistic Regression: ', lr_result.mean())

rf_model = RandomForestClassifier()
rf_result = cross_val_score(rf_model, X_trn_pseknc, y_trn_pseknc, cv=kfold)
print('Random Forest: ', rf_result.mean())

svm_model = SVC()
svm_result = cross_val_score(svm_model, X_trn_pseknc, y_trn_pseknc, cv=kfold)
print('Support Vector Machine: ', svm_result.mean())

ab_model = AdaBoostClassifier()
ab_result = cross_val_score(ab_model, X_trn_pseknc, y_trn_pseknc, cv=kfold)
print('AdaBoost: ', ab_result.mean())

xgb_model = XGBClassifier()
xgb_result = cross_val_score(xgb_model, X_trn_pseknc, y_trn_pseknc, cv=kfold)
print('XGBoost: ', xgb_result.mean())


# ### Independent Test

# In[17]:


ml_model = XGBClassifier()
ml_model.fit(X_trn_kmer, y_trn_kmer)
true_labels = np.asarray(y_tst_kmer)
predictions = ml_model.predict(X_tst_kmer)
print(confusion_matrix(true_labels, predictions))
print(accuracy_score(true_labels, predictions))


# ## ROC Curves

# ### Feature Comparison

# In[10]:


from sklearn.model_selection import train_test_split
def calculate_ROC(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
     
    # Train SVM classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Probability
    pred_prob = model.predict_proba(X_test)

    #GET ROC DATA
    fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


# In[23]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
def calculate_PRC(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
     
    # Train SVM classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Probability
    pred_prob = model.predict_proba(X_test)
    
    # predict class values
    yhat = model.predict(X_test)

    #GET ROC DATA
    precision, recall, _ = precision_recall_curve(y_test, pred_prob[:,1])
    f1, prc_auc = f1_score(y_test, yhat), auc(recall, precision)
    
    return recall, precision, f1


# In[13]:


fpr_kmer, tpr_kmer, roc_auc_kmer = calculate_ROC(X_trn_kmer, y_trn_kmer)
fpr_psednc, tpr_psednc, roc_auc_psednc = calculate_ROC(X_trn_psednc, y_trn_psednc)
fpr_pseknc, tpr_pseknc, roc_auc_pseknc = calculate_ROC(X_trn_pseknc, y_trn_pseknc)
fpr_DAC, tpr_DAC, roc_auc_DAC = calculate_ROC(X_trn_DAC, y_trn_DAC)
fpr_DACC, tpr_DACC, roc_auc_DACC = calculate_ROC(X_trn_DACC, y_trn_DACC)
fpr_DCC, tpr_DCC, roc_auc_DCC = calculate_ROC(X_trn_DCC, y_trn_DCC)
fpr_TAC, tpr_TAC, roc_auc_TAC = calculate_ROC(X_trn_TAC, y_trn_TAC)
fpr_TACC, tpr_TACC, roc_auc_TACC = calculate_ROC(X_trn_TACC, y_trn_TACC)
fpr_TCC, tpr_TCC, roc_auc_TCC = calculate_ROC(X_trn_TCC, y_trn_TCC)


# In[71]:


# Plot ROC Curve
fig_roc_feat = plt.figure(figsize=(15,15))
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
#plt.title('SVM Classifier ROC', fontsize=20)
plt.plot(fpr_kmer, tpr_kmer, lw=4, label='kmer: AUC = %0.3f' % roc_auc_kmer)
plt.plot(fpr_DAC, tpr_DAC, lw=4, label='DAC: AUC = %0.3f' % roc_auc_DAC)
plt.plot(fpr_DACC, tpr_DACC, lw=4, label='DACC: AUC = %0.3f' % roc_auc_DACC)
plt.plot(fpr_DCC, tpr_DCC, lw=4, label='DCC: AUC = %0.3f' % roc_auc_DCC)
plt.plot(fpr_TAC, tpr_TAC, lw=4, label='TAC: AUC = %0.3f' % roc_auc_TAC)
plt.plot(fpr_TACC, tpr_TACC, lw=4, label='TACC: AUC = %0.3f' % roc_auc_TACC)
plt.plot(fpr_TCC, tpr_TCC, lw=4, label='TCC: AUC = %0.3f' % roc_auc_TCC)
plt.plot(fpr_psednc, tpr_psednc, lw=4, label='PseDNC: AUC = %0.3f' % roc_auc_psednc)
plt.plot(fpr_pseknc, tpr_pseknc, lw=4, label='PseKNC: AUC = %0.3f' % roc_auc_pseknc)

plt.plot(fpr_cnn, tpr_cnn, lw=4, label='BERT: AUC = %0.3f' % roc_auc_cnn)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="lower right", prop={'size': 20})
plt.show()


# In[73]:


fig_roc_feat.savefig(os.path.join('new_fig', '6ma_ind_roc_features.png'), dpi=300, bbox_inches='tight')


# ### Precision-Recall Curve

# In[129]:


recall_kmer, precision_kmer, prc_auc_kmer = calculate_PRC(X_trn_kmer, y_trn_kmer)
recall_psednc, precision_psednc, prc_auc_psednc = calculate_PRC(X_trn_psednc, y_trn_psednc)
recall_pseknc, precision_pseknc, prc_auc_pseknc = calculate_PRC(X_trn_pseknc, y_trn_pseknc)
recall_bert, precision_bert, prc_auc_bert = calculate_PRC(X_trn_bert, y_trn_bert)


# In[130]:


# Plot PR Curve
fig = plt.figure(figsize=(15,15))
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
#plt.title('SVM Classifier ROC', fontsize=20)
plt.plot(recall_kmer, precision_kmer, marker='.', lw=4, label='kmer: F1-score = %0.2f)' % prc_auc_kmer)
plt.plot(recall_psednc, precision_psednc, marker=',', lw=4, label='PseDNC: F1-score = %0.2f)' % prc_auc_psednc)
plt.plot(recall_pseknc, precision_pseknc, marker='o', lw=4, label='PseKNC: F1-score = %0.2f)' % prc_auc_pseknc)
plt.plot(recall_bert, precision_bert, marker='v', lw=4, label='BERT: F1-score = %0.2f)' % prc_auc_bert)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="upper right", prop={'size': 20})
plt.show()


# ### Classifier Comparison

# In[15]:


from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

def calculate_ROC_classifier(X, y, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    if classifier=='lr':
        model = LogisticRegression()
    if classifier=='xgb':
        model = XGBClassifier()
    if classifier=='svm':
        model = SVC(probability=True)
    if classifier=='ab':
        model = AdaBoostClassifier()
    if classifier=='rf':
        model = RandomForestClassifier()
        
    model.fit(X_train, y_train)

    # Probability
    pred_prob = model.predict_proba(X_test)

    #GET ROC DATA
    fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


# In[16]:


def calculate_ROC_CNN(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = d_cnn_model()
    
    trn_new = np.asarray(X_train)
    tst_new = np.asarray(X_test)   
    ## evaluate the model
    model.fit(trn_new.reshape(len(trn_new),num_features,1), 
              np_utils.to_categorical(y_train,nb_classes), 
              epochs=num_epochs, batch_size=10, verbose=0, class_weight='auto')
    pred_prob = model.predict_proba(tst_new.reshape(len(tst_new),num_features,1))
#     print(pred_prob)
    
    #GET ROC DATA
    fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


# In[55]:


# fpr_lr, tpr_lr, roc_auc_lr = calculate_ROC_classifier(X_trn, y_trn, 'lr')
# fpr_ab, tpr_ab, roc_auc_ab = calculate_ROC_classifier(X_trn, y_trn, 'ab')
# fpr_rf, tpr_rf, roc_auc_rf = calculate_ROC_classifier(X_trn, y_trn, 'rf')
# fpr_svm, tpr_svm, roc_auc_svm = calculate_ROC_classifier(X_trn, y_trn, 'svm')
# fpr_xgb, tpr_xgb, roc_auc_xgb = calculate_ROC_classifier(X_trn, y_trn, 'xgb')


# In[62]:


fpr_cnn, tpr_cnn, roc_auc_cnn = calculate_ROC_CNN(X_trn, y_trn)


# In[68]:


# Plot ROC Curve
fig = plt.figure(figsize=(15,15))
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# plt.plot(fpr_lr, tpr_lr, marker='.',lw=4, label='Logistic Regression: AUC = %0.3f' % roc_auc_lr)
plt.plot(fpr_rf, tpr_rf, marker='o', ls='-',lw=4, label='Random Forest: AUC = %0.3f' % roc_auc_rf)
plt.plot(fpr_svm, tpr_svm, marker='v', ls='-',lw=4, label='Support Vector Machine: AUC = %0.3f' % roc_auc_svm)
plt.plot(fpr_ab, tpr_ab, marker=',', ls='-',lw=4, label='AdaBoost: AUC = %0.3f' % roc_auc_ab)
plt.plot(fpr_xgb, tpr_xgb, marker='v', ls='-',lw=4, label='XGBoost: AUC = %0.3f' % roc_auc_xgb)
plt.plot(fpr_cnn, tpr_cnn, marker='v', ls='-',lw=4, label='Convolutional Neural Network: AUC = %0.3f' % roc_auc_cnn)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="lower right", prop={'size': 20})
plt.show()


# In[70]:


fig.savefig(os.path.join('new_fig', '6ma_ind_roc_classifiers.png'), dpi=300, bbox_inches='tight')


# ## Independent test comparison

# In[17]:


def calculate_ROC_features_independent(X_train, y_train, X_test, y_test):
     
    # Train SVM classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Probability
    pred_prob = model.predict_proba(X_test)

    #GET ROC DATA
    fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


# In[18]:


def calculate_ROC_classifiers_independent(X_train, y_train, X_test, y_test, classifier):
    
    if classifier=='lr':
        model = LogisticRegression()
    if classifier=='xgb':
        model = XGBClassifier()
    if classifier=='svm':
        model = SVC(probability=True)
    if classifier=='ab':
        model = AdaBoostClassifier()
    if classifier=='rf':
        model = RandomForestClassifier()
        
    model.fit(X_train, y_train)

    # Probability
    pred_prob = model.predict_proba(X_test)

    #GET ROC DATA
    fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


# In[44]:


def calculate_ROC_CNN_independent(X_train, y_train, X_test, y_test):
    model = d_cnn_model()
    
    trn_new = np.asarray(X_train)
    tst_new = np.asarray(X_test)   
    ## evaluate the model
    model.fit(trn_new.reshape(len(trn_new),num_features,1),
              utils.to_categorical(y_train, num_classes=nb_classes), 
              epochs=num_epochs, 
              batch_size=10, 
              verbose=0)
    pred_prob = model.predict(tst_new.reshape(len(tst_new),num_features,1))
#     print(pred_prob)
    
    #GET ROC DATA
    fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


# In[74]:


fpr_kmer, tpr_kmer, roc_auc_kmer = calculate_ROC_features_independent(X_trn_kmer, y_trn_kmer, X_tst_kmer, y_tst_kmer)
fpr_psednc, tpr_psednc, roc_auc_psednc = calculate_ROC_features_independent(X_trn_psednc, y_trn_psednc, X_tst_psednc, y_tst_psednc)
fpr_pseknc, tpr_pseknc, roc_auc_pseknc = calculate_ROC_features_independent(X_trn_pseknc, y_trn_pseknc, X_tst_pseknc, y_tst_pseknc)
fpr_DAC, tpr_DAC, roc_auc_DAC = calculate_ROC_features_independent(X_trn_DAC, y_trn_DAC, X_tst_DAC, y_tst_DAC)
fpr_DACC, tpr_DACC, roc_auc_DACC = calculate_ROC_features_independent(X_trn_DACC, y_trn_DACC, X_tst_DACC, y_tst_DACC)
fpr_DCC, tpr_DCC, roc_auc_DCC = calculate_ROC_features_independent(X_trn_DCC, y_trn_DCC, X_tst_DCC, y_tst_DCC)
fpr_TAC, tpr_TAC, roc_auc_TAC = calculate_ROC_features_independent(X_trn_TAC, y_trn_TAC, X_tst_TAC, y_tst_TAC)
fpr_TACC, tpr_TACC, roc_auc_TACC = calculate_ROC_features_independent(X_trn_TACC, y_trn_TACC, X_tst_TACC, y_tst_TACC)
fpr_TCC, tpr_TCC, roc_auc_TCC = calculate_ROC_features_independent(X_trn_TCC, y_trn_TCC, X_tst_TCC, y_tst_TCC)


# In[67]:


fpr_cnn, tpr_cnn, roc_auc_cnn = calculate_ROC_CNN_independent(X_trn, y_trn, X_tst_bert, y_tst_bert)


# In[41]:


X_trn = X_trn_bert
y_trn = y_trn_bert


# In[42]:


fpr_lr, tpr_lr, roc_auc_lr = calculate_ROC_classifiers_independent(X_trn, y_trn, X_tst_bert, y_tst_bert, 'lr')
fpr_ab, tpr_ab, roc_auc_ab = calculate_ROC_classifiers_independent(X_trn, y_trn, X_tst_bert, y_tst_bert, 'ab')
fpr_rf, tpr_rf, roc_auc_rf = calculate_ROC_classifiers_independent(X_trn, y_trn, X_tst_bert, y_tst_bert, 'rf')
fpr_svm, tpr_svm, roc_auc_svm = calculate_ROC_classifiers_independent(X_trn, y_trn, X_tst_bert, y_tst_bert, 'svm')
fpr_xgb, tpr_xgb, roc_auc_xgb = calculate_ROC_classifiers_independent(X_trn, y_trn, X_tst_bert, y_tst_bert, 'xgb')


# ## Hyperparameters Tuning

# In[23]:


# A parameter grid for XGBoost
xgb_params = {
    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
    'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
        }

rf_params = {
    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
        }

svc_params = {
    'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1], 'kernel':['rbf','linear']
}


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
param_comb = 200

skf = StratifiedKFold(n_splits=5, shuffle = True)

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_params, n_iter=param_comb, scoring='accuracy', n_jobs=4, 
                                   cv=skf.split(X_trn, y_trn), 
                                   verbose=3)

# Here we go
random_search.fit(X_trn, y_trn)


# ## Deep Learning

# In[8]:


from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import utils


# In[75]:


num_features = 768
num_epochs = 20
nb_classes = 2
nb_kernels = 3
nb_pools = 2


# In[2]:


def dnn_model():
    model = Sequential()
    model.add(Dense(100, input_dim = X_trn.shape[1], kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    
    model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    return model


# In[83]:


def d_cnn_model(input_length):
    model = Sequential()

    model.add(Dropout(0.2, input_shape=(input_length,1)))
    model.add(Conv1D(32, 3, activation='relu'))
    # model.add(Conv1D(32, 3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 3, activation='relu'))
    # # model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 3, activation='relu'))
    # # model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))
    
#     model.add(Conv1D(256, 3, activation='relu'))
#     # # model.add(Dropout(0.5))
#     model.add(MaxPooling1D(2))
    
#     model.add(Conv1D(512, 3, activation='relu'))
#     # # model.add(Dropout(0.5))
#     model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


# In[10]:


epochs = 100
batch_size = 32
learning_rate = 0.01

kfold = StratifiedKFold(n_splits = 5, shuffle= True, random_state=42)
cvscores=[]
for train_idx, test_idx in kfold.split(X_trn,y_trn):
    train_X, test_X, train_y, test_y = X_trn.iloc[train_idx], X_trn.iloc[test_idx], y_trn.iloc[train_idx], y_trn.iloc[test_idx]
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    model = build()
    model.fit(train_X, train_y, epochs = epochs, batch_size = batch_size,validation_data = (test_X, test_y), verbose = 0)
    scores = model.evaluate(test_X, test_y, verbose = 0)
    print('%s: %.2f%%'%(model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)
    
print('Mean accuracy: %.2f%% (S.D.: %.2f%%)'%(np.mean(cvscores), np.std(cvscores)))


# ### 1D CNN Training

# In[ ]:


# CNN
_10fold = StratifiedKFold(n_splits=10, shuffle=True)
for train, test in _10fold.split(X_trn, y_trn):
    model = d_cnn_model()
    trn_new = np.asarray(X_trn.iloc[train])
    tst_new = np.asarray(X_trn.iloc[test])   
    ## evaluate the model
    model.fit(trn_new.reshape(len(trn_new),num_features,1), 
              np_utils.to_categorical(y_trn.iloc[train],nb_classes), 
              epochs=num_epochs, batch_size=10, verbose=0, class_weight='auto')
    #prediction
    predictions = model.predict_classes(tst_new.reshape(len(tst_new),num_features,1))
    true_labels_cv = np.asarray(y_trn.iloc[test])
    print('CV: ', confusion_matrix(true_labels_cv, predictions))
#     print(classification_report(true_labels_cv, predictions))


# In[81]:


trn_new.shape[1]


# In[87]:


# Independent Test
trn_new = np.asarray(X_trn_bert)
tst_new = np.asarray(X_tst_bert)
final_model = d_cnn_model(trn_new.shape[1])

final_model.fit(trn_new.reshape(len(trn_new),trn_new.shape[1],1),
          to_categorical(y_trn_kmer,nb_classes),
          epochs=num_epochs, batch_size=10, verbose=0)

predictions = final_model.predict_classes(tst_new.reshape(len(tst_new),tst_new.shape[1],1))
true_labels_ind = np.asarray(y_tst_kmer)
print('IND: ', confusion_matrix(true_labels_ind, predictions))


# In[ ]:


# Comparison among different features
for key, value in feat_dict.items():
    TP = FP = TN = FN = 0
    acc_cv_scores = []
    auc_cv_scores = []
    svm_model = d_cnn_model()
    
    ## evaluate the model
    svm_model.fit(value.iloc[train], y_trn.iloc[train])
    # evaluate the model
    true_labels = np.asarray(y_trn.iloc[test])
    predictions = svm_model.predict(value.iloc[test])
    acc_cv_scores.append(accuracy_score(true_labels, predictions))
    # print(confusion_matrix(true_labels, predictions))
    newTN, newFP, newFN, newTP = confusion_matrix(true_labels,predictions).ravel()
    TP += newTP
    FN += newFN
    FP += newFP
    TN += newTN
    pred_prob = svm_model.predict_proba(value.iloc[test])
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_prob[:,1], pos_label=1)
    auc_cv_scores.append(metrics.auc(fpr, tpr))
    # print('AUC = ', metrics.auc(fpr, tpr))
    # print('AUC = ', round(metrics.roc_auc_score(true_labels, predictions)*100,2))

    print('\nFeature: ', key)
    print('Accuracy = ', np.mean(acc_cv_scores))
    print('TP = %s, FP = %s, TN = %s, FN = %s' % (TP, FP, TN, FN))
    print('AUC = ', np.mean(auc_cv_scores))


# ## Plot model history

# In[89]:


trn_new = np.asarray(X_trn_bert)
tst_new = np.asarray(X_tst_bert)
# Fit the model
history = final_model.fit(trn_new.reshape(len(trn_new),trn_new.shape[1],1),
                    to_categorical(y_trn,nb_classes),
                    validation_split=0.25, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())


# In[60]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[61]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[64]:


# list all data in history
print(final_model.history.keys())
# summarize history for accuracy
plt.plot(final_model.history['accuracy'])
plt.plot(final_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Recursive Feature Elimination

# In[15]:


from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=RandomForestClassifier(), step=1, cv=StratifiedKFold(3),
              scoring='accuracy')
rfecv.fit(X_trn_bert, y_trn_bert)

print("Optimal number of features : %d" % rfecv.n_features_)


# ## EDA

# In[ ]:




