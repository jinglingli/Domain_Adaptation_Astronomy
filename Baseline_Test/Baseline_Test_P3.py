
# coding: utf-8

# The Baseline test used random forest classifier to get the baseline classification results when using a) target data as training data, and b) source data+target data as training data.  
# 
# The following ipython notebook (compatible with python 2.7+, need modifications for python 3) needs the following input files in the specified input format to get the baseline performance:
# 
# -feature_file: 
# A comma separated file which contains corresponding numbering and names of the extracted features from the light curve in the data file. The first column denotes the numbering of each feature and the second column denotes the name of each feature. 'config.dat' is the feature file we used for this research.
# 
# -selected_features:
# A list which contains the numbering of the features we selected for this research
# 
# -Class_list:
# A tab separated file which contains all class types appeared in the data file. The first column denotes the name of each class type and the fourth column denotes the numbering of each class type, which are correspondent to the last column of each data file. For example, if one row has 'EW' in the first column and 4 in the fourth column in the file Class_list, an object with class type 1 is an 'EW' object. 
# 
# -Selected_Class_names:
# A list which contains the names of selected class types we used for this research
# 
# -Data files
# In this research, we used csdr2('CSDR2_lc_data.csv'), ptfr('R_PTF_lc_features.csv'), and lineardb('Linear_lc_over40.csv') as our datasets. Each of the dataset has the following format:
# 
# The headerline includes the names of all features associated with this dataset along with the class type for each object;
# 
# Each line (except the header) denotes one data entry and contains the information of all features for one object;
# 
# The last column of each dataset denotes the class types for each object, the numbering of the class types should coresspond to the indices in Class_list

# In[1]:

from __future__ import division
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import cross_validation as cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import setp
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from collections import Counter
from sklearn.metrics import confusion_matrix
from itertools import compress
import pickle


# More details about the baseline test
# * no sampling methods, neither SMOTE nor under as we have ample data
# * select 400~500 from each class from each survey to form the domain data
# * eleminate part of the source data in training which has the same ID as the cross-validation data
# * save 10% objects from each class for testing
# * Use K-fold cross-validation to test for base results
# * plot with error bar

# In[2]:

#creat feature labels for data
def add_Names(filename, selected_features):
    # Start with an empty dictionary
    stats = {}
    for line in open(filename):
        # Split the feature file with delimiter ','; key is the feature number and value is feature name
        line = line.replace("'", "")
        temp = line.rstrip().split(',')
        stats[temp[0]] = temp[1]
    features = []
    for feature in selected_features:
        features.append(stats[str(feature)])
    features.append(class_label)
    return features

#Select data based on objects' labels (subset for certain classes)
def selectdata(data, Selected_Class):
    dataF = pd.DataFrame()
    i = 0
    for c in Selected_Class:
        dataF =  dataF.append(data[data[class_label] == c])
        #dataF.loc[data[class_label] == c,class_label] = i
        i+=1
    return dataF

def test_train_label(X, split, offset):
    len_feature = len(X.columns) - offset
    train, test = cv.train_test_split(X, train_size=(split/100))
    X_train = train.iloc[:,range(0,len_feature)]
    y_train= train[class_label]
    X_test = test.iloc[:,range(0,len_feature)]
    y_test = test[class_label]
    return X_train,y_train,X_test,y_test

def normalize_data_with_label(data):
    df = data.iloc[:,range(0,len_feature)]
    data_norm = (df - df.mean()) / (df.max() - df.min())
    data_norm[class_label] = data[class_label]
    return data_norm

#minus mean and divided by standard deviation
def normalize_data_with_label2(data):
    df = data.iloc[:,range(0,len_feature)]
    data_norm = (df - df.mean()) / df.std()
    data_norm[class_label] = data[class_label]
    return data_norm

def sample_wo_replacement(data, size, Selected_Class):
    o_data = pd.DataFrame()
    for c in Selected_Class:
        temp = data[data[class_label] == c]
        class_size = len(temp)
        if size > class_size:
            indexes = np.random.choice(temp.index, size-class_size, replace=True)
            temp = temp.append(temp.ix[indexes])
        else:
            indexes = np.random.choice(temp.index, size, replace=False)
            temp = temp.ix[indexes]
        o_data = o_data.append(temp)
    return o_data

def sample_wo_replacement_by_ratio(data, ratio, Selected_Class):
    o_data = pd.DataFrame()
    for c in Selected_Class:
        temp = data[data[class_label] == c]
        class_size = len(temp)    
        if ratio > 1:
            indexes = np.random.choice(temp.index, class_size*(ratio-1), replace=True)
            temp = temp.append(temp.ix[indexes])
        else:
            indexes = np.random.choice(temp.index, class_size*ratio, replace=False)
            temp = temp.ix[indexes]
        o_data = o_data.append(temp)
    return o_data

def findSmallestClassSize(data, Selected_Class):
    smallest_size = 9999999999
    for c in Selected_Class:
        temp = data[data[class_label] == c]
        class_size = len(temp)
        if smallest_size > class_size:
            smallest_size = class_size
    return smallest_size


# In[3]:

def Domain_Adaptation23_modified(S_data, T_data, Clf, target_smaller_class, source_smaller_class, checkBoth):
    Learning_accuracies = []
    Feature_importance = []
    Learning_accuracies_TT = []
    Feature_importance_TT = []
    #select n number of data from each class in target domain for cross-validation
    Y_tests = []
    Y_preds = []
    Y_preds_TT = []
    Y_indexes = []
    for split in range(start_size,end_size,bins):
        Y_test = []
        Y_pred = []
        Y_pred_TT = []
        Y_index = []
        learn_accuracy = []
        learn_accuracy_TT = []
        importances = []
        importances_TT = []
        #select size objects from each class for training
        size = ((split/100)*target_smaller_class)
        for i in range(0, 10):
            Source_train = sample_wo_replacement(S_data, source_smaller_class, Selected_Class)
            #perform iter_time1 selecting Target_train
            for j in range(0,iter_time1):
                scores = []
                scores_TT = []
                #size: the size of a single class, so in total, target data for training will be size*num_of_classes
                Target_train = sample_wo_replacement(T_data, size, Selected_Class)         
                #perform iter_time2 slecting Test_target
                for k in range(0,iter_time2):
                    #include objects that are in the class
                    Test_target = (T_data[~T_data.index.isin(Target_train.index.values)]).sample(frac=0.1)
                    X_test = np.array(Test_target.ix[:,range(0,len_feature)])
                    y_test = Test_target[class_label].values
                    y_index = Test_target.index.values
                    
                    Train_data = Target_train.append(Source_train)
                    X_train = np.array(Train_data.ix[:,range(0,len_feature)])
                    y_train = Train_data[class_label].values
                    Clf.fit(X_train, y_train)
                    scores.append(Clf.score(X_test, y_test))
                    Y_test.extend(y_test)
                    Y_pred.extend(Clf.predict(X_test))
                    Y_index.extend(y_index)
                    if (type(Clf) == type(rfc)):
                        importances.append(Clf.feature_importances_)
                    if checkBoth:
                        X_train = np.array(Target_train.ix[:,range(0,len_feature)])
                        y_train = Target_train[class_label].values
                        #y_train = y_train.astype(int)
                        Clf.fit(X_train, y_train)
                        scores_TT.append(Clf.score(X_test, y_test))
                        Y_pred_TT.extend(Clf.predict(X_test))
                        if (type(Clf) == type(rfc)):
                            importances_TT.append(Clf.feature_importances_)
                learn_accuracy.append(np.mean(scores))
                learn_accuracy_TT.append(np.mean(scores_TT))
        if (type(Clf) == type(rfc)):
            ave_f_importance =  np.mean(np.array(importances), axis=0)
            Feature_importance.append(ave_f_importance)
            ave_f_importance_TT =  np.mean(np.array(importances_TT), axis=0)
            Feature_importance_TT.append(ave_f_importance_TT)
        Learning_accuracies.append(np.mean(learn_accuracy))
        Y_tests.append(Y_test)
        Learning_accuracies_TT.append(np.mean(learn_accuracy_TT))
        Y_preds_TT.append(Y_pred_TT)
        Y_preds.append(Y_pred)
        Y_indexes.append(Y_index)
    return Learning_accuracies, Learning_accuracies_TT, Clf, Feature_importance, Feature_importance_TT, Y_tests, Y_preds, Y_preds_TT, Y_indexes


# In[4]:

start_size = 5 #starting ratio that denotes the % of the target data used as training set
end_size = 50 #ending ratio that denotes the % of the target data used as training set
bins = 5 
runs = 4 #number of runs for each baseline test
iter_time1 = 10 #number of iterations where each iteration samples a balanced training set from the target domain
iter_time2 = 10 #number of iterations where each iteration samples a testing set from the target domain


# In[ ]:

#creating feature labels
feature_file = 'config.dat'
class_label = "class"
selected_features = list(range(1,22))
selected_features.remove(15)
len_feature = len(selected_features)
labels = add_Names(feature_file,selected_features)

#creating selected class types
classes = {}
Class_list = 'Class_list'
for line in open(Class_list):
    # Split the config.dat file with delimiter ','; key is the feature number and value is feature name
    line = line.replace("'", "")
    temp = line.rstrip().split('\t')
    classes[temp[0]] = temp[3]
Selected_Class_names = ["EW","EA","RRab","RRc","RRd","RS CVn"]  
Selected_Class = [int(classes[x]) for x in Selected_Class_names]

rfc = RandomForestClassifier(class_weight='auto')
clf = rfc

#Reading data
csdr2 = pd.read_csv('CSDR2_lc_data.csv')
ptfr = pd.read_csv('R_PTF_lc_features.csv')
lineardb = pd.read_csv('Linear_lc_over40.csv')
test_smaller_class = len(ptfr[ptfr[class_label]==6]) 
crts = csdr2

'''Domain Adaptation learning curve for T-2-T and S+T-2-T'''
Learning_curves = []
labels = []
Y_Tests = []
Y_Preds_TT = []
Y_Preds = []
Y_indexes = []
for run in range(0,runs):
    print ("run %d" %run)
    Learning_curve = []
    for S_data, T_data, s_name, t_name, checkBoth in [(lineardb, crts, "Lineardb", "CRTS", True),
                                                      (crts, ptfr, "CRTS", "PTF(r)", True),
                                                      (ptfr, crts, "PTF(r)", "CRTS", False),
                                                      (crts, lineardb, "CRTS", "Lineardb", True), 
                                                      (ptfr, lineardb, "PTF(r)", "Lineardb", False), 
                                                      (lineardb, ptfr, "Lineardb", "PTF(r)",False)]:
        S_data = normalize_data_with_label2(S_data)
        T_data = normalize_data_with_label2(T_data)
        target_smaller_class = findSmallestClassSize(T_data, Selected_Class)
        source_smaller_class = findSmallestClassSize(S_data, Selected_Class)
        Learning_accuracies, Learning_accuracies_TT, Clf, Feature_importance, Feature_importance_TT, Y_tests, Y_preds, Y_preds_TT, Y_index = Domain_Adaptation23_modified(
            S_data, T_data, clf, target_smaller_class, source_smaller_class, checkBoth)
        pickle.dump([Learning_accuracies, Feature_importance], open("S+T to T %s-2-%s_run%d.p" % (s_name,t_name, run), "wb" ))
        Learning_curve.append(Learning_accuracies)
        labels.append("S+T to T %s-2-%s" % (s_name,t_name))
        
        if checkBoth:
            Learning_curve.append(Learning_accuracies_TT)
            labels.append("T to T %s" % t_name)
            Y_Preds_TT.append(Y_preds_TT)
            pickle.dump([Learning_accuracies_TT, Feature_importance_TT], open("T to T %s_run%d.p" % (t_name,run), "wb" ))
        Y_Tests.append(Y_tests)
        Y_Preds.append(Y_preds)
        Y_indexes.append(Y_index)   
    Learning_curves.append(np.array(Learning_curve))
fig = plt.figure(figsize=(18.75,10))
ax = fig.add_subplot(111)
curve_plot = np.mean(Learning_curves,axis=0)
pickle.dump([Learning_curves, labels], open("Learning_curves_w_labels.p", "wb" ))
std_err = sem(Learning_curves)
for index in range(0, curve_plot.shape[0]):
    #ax.plot(range(start_size,end_size,bins), Learning_accuracies, "s-",
    #            label=labels[index],marker = Line2D.filled_markers[index])
    ax.errorbar(range(start_size,end_size,bins), curve_plot[index], yerr=std_err[index], 
                label=labels[index], marker = Line2D.filled_markers[index])

ax.set_title("Average acurracies for S+T-to-T and T-to-T (three surveys)",fontsize=24)
ax.set_xlabel("Target data training percentage %",fontsize=18)
ax.set_ylabel("Average accuracy",fontsize=18)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
# Put a legend to the right of the current axis
ax.legend(loc='upper left', bbox_to_anchor=(1, 0.6),fontsize=18)
plt.savefig('three_surveys_RF_DA2_3.png')


# ---------------------------------------K_fold_cross_validation baseline tests for in-domain accuracy---------------------------------------

# In[14]:

def K_fold_cross_validation(K_fold, crts, ptfr, lineardb):
    Y_Tests = []
    Y_Preds = []
    print ("%d_fold cross validation: " %K_fold)
    for data, name in [(crts,"crts"), (ptfr,"ptfr"), (lineardb,"lineardb")]:
        data = normalize_data_with_label2(data)
        Y_test = []
        Y_pred = []
        Indexes = []
        test_smaller_class = len(data[data[class_label]==6])
        for c in Selected_Class:
            temp =  data[data[class_label] == c]
            len_size = test_smaller_class//K_fold #len(temp) or to get a balanced data: test_smaller_class 
            chunks = []
            for i in range(0, K_fold):   
                rows = temp.sample(len_size).index.values
                chunks.append(rows)
                temp = temp.drop(rows)
            Indexes.append(chunks)

        #Every row in Indexes is the data for class c that are splited into K chuncks
        class_index = []
        for i in range(0, K_fold):
            indexes_split = []
            for c in range(0,len(Indexes)):
                indexes_split.extend(Indexes[c][i])
            class_index.append(indexes_split)

        #K_fold cross validation
        scores = []
        for class_for_test in range(0, K_fold):
            temp_index = []
            for i in range(0, K_fold):
                if(i is not class_for_test):
                    temp_index.extend(class_index[i])
            train = data[data.index.isin(temp_index)]
            test = data[data.index.isin(class_index[class_for_test])]
            rfc.fit(train.iloc[:,range(0,len_feature)], train[class_label])
            Y_pred.extend(rfc.predict(test.iloc[:,range(0,len_feature)]))
            Y_test.extend(test[class_label])        
            scores.append(rfc.score(test.iloc[:,range(0,len_feature)], test[class_label]))
        print ("average accuracy for %s is %f" % (name,np.mean(scores)))
        Y_Tests.append(Y_test)       
        Y_Preds.append(Y_pred)
    return Y_Tests, Y_Preds

def plot_confusion_matrix(cm, title, Selected_Class, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(Selected_Class))
    plt.xticks(tick_marks, Selected_Class, rotation=45)
    plt.yticks(tick_marks, Selected_Class)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:

#confusion matrix
'''
names = ['crts','ptfr','lineardb']
K = 2
Y_Test, Y_Pred = K_fold_cross_validation(K, crts, ptfr, lineardb)
for i in range(len(Y_Test)):
    cm = confusion_matrix(Y_Test[i], Y_Pred[i])
    np.set_printoptions(precision=3)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized Confusion matrix for %s under %d-fold cv' %(names[i], K))
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, names[i], Selected_Class)
'''

