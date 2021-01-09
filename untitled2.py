# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 11:47:46 2020

@author: 33754
"""


import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.palettes import Spectral6, Spectral5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *
from plotnine.data import *

from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from scipy.stats import pearsonr, spearmanr, chi2_contingency, ttest_ind, mannwhitneyu
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import NeighborhoodComponentsAnalysis,KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import pandas as pd
data= pd.read_csv('drug.csv')
for i in range(19):
    
    string=data.columns[i+13]    
    data.loc[data[string]=="CL0",string]=0
    for h in range(1,7):
        stringer="CL"+str(h)
        data.loc[data[string]==stringer,string]=1
data=data[data["semer"]==0]
dat_dd= data.copy()
dat_dd['age']=['18-24' if age == -0.95197 else 
           '25-34' if age == -0.07854 else 
           '35-44' if age == 0.49788 else 
           '45-54' if age == 1.09449 else 
           '55-64' if age == 1.82213 else 
           '65+' 
           for age in dat_dd['age']]
dat_dd['age']=['20' if age == -0.95197 else 
       '30' if age == -0.07854 else 
       '40' if age == 0.49788 else 
       '50' if age == 1.09449 else 
       '60' if age == 1.82213 else 
       '80' 
       for age in dat_dd['age']]

liste=[]

for z in range(dat_dd.shape[0]):    
    
    nb_drugs=0
    for i in range(19):
        string=data.columns[i+13]
        nb_drugs+=data.iloc[z][string]
    liste.append(nb_drugs)

for i in range( len(liste)):
    if liste[i] in [0,1,2]:
        liste[i]=0
    if liste[i] in [3,4,5]:
        liste[i]=0
    if liste[i] in [6,7,8]:
        liste[i]=0
    if liste[i] in [9,10,11]:
        liste[i]=1
    if liste[i] in [12,13,14]:
        liste[i]=1
    if liste[i] in [15,16,17,18]:
        liste[i]=1
dat_dd.insert(1,'totalDrugTested2',liste)
X,Y= dat_dd[["Nscore","Escore","Oscore","Ascore","Cscore","impulsive","SS","age"]],dat_dd['totalDrugTested2']
Xtraining, Xtest, Ytraining, Ytest = train_test_split(X, Y, test_size=0.2)
Modellog = LogisticRegression(max_iter=1000)
Modellog.fit(Xtraining, Ytraining)
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    final_features = np.reshape(final_features,(1,8))
    prediction = Modellog.predict(final_features)
    

    return render_template('index.html', prediction_text='{} \n if you got "1" then you have tested more than average. if you got 0 then you tested less than average'.format(prediction))
    #return str(final_features.shape)



if __name__ == "__main__":
    app.run(debug=True)