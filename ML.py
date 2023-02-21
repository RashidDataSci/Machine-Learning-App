# import libraries
from datetime import date
import imp
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# title of app
st.write('''
# Explore different ML Models and Data sets
Which is the best for them?
''')

# box for different data sets
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# box for different classifier
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN','SVM', 'Random Forest')
)

# define a function for load data set
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name =='Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y 

# call the function and give name x, y to dataset_name
x, y = get_dataset(dataset_name)

# print the shape of data set 
st.write('Shape of dataset:', x.shape) 
st.write('Number of classes:', len(np.unique(y)))

# next we put different parameter in their classifiers
def add_parameter_ui(classifier_name):
    params = dict() # create an empty dictionary
    if classifier_name == 'SVM':
        c = st.sidebar.slider('c', 0.01, 10.0)
        params['c'] = c # its the degree of correct classification
    elif classifier_name == 'KNN':
        k = st.sidebar.slider('k',1,15)
        params['k'] = k # its the number of nearest neighber
    else:
        max_depth = st.sidebar.slider('max_depth',2, 15)
        params['max_depth'] = max_depth # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators # number of tree
    return params

# now we call the function and equal with their params
params = add_parameter_ui(classifier_name)

# now we build classifier on the base of classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['c'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], random_state=1234)
    return clf

# now we call this function and equal with clf
clf = get_classifier(classifier_name, params)

# split our data in x, y train and test by 80/20 ratio
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1234)

# training our classifier with data set
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# after checking model accuracy print the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', accuracy)

#### Plot Dataset ####
# draw on 2 dimension with pca
pca = PCA(2)
x_projected = pca.fit_transform(x)

# slice the data 0 or 1 dimension
x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)
