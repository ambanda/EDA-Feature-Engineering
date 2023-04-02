import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
df=pd.read_csv("D:\HOME\productivity\Feb2019.csv")
print(df.head(5))
#Predictive Modeling overview
#introduction to data
#assign outcome(label) as binary
df['label']= [0 if x=='' else 1 for x in df['label']]
#assign X as dataframe of features and y as label
x= df.drop('label',1)
y=df.label 

#Basic data cleaning
#dealing with data types
#use get_dummies in pandas or OneHotEncoder in scikit
print(pd.get_dummies(x['']).head())
#decide which categorical variables you want to use in model
for col_name in x.columns:
    if x[col_name].dtypes == 'object':
        unique_cat = len(x[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name,unique_cat=unique_cat))
#choose specific column to find out how many unique categories
print(x['specific column'].value_count().sortvalues(ascending=False).head())
#bucketlist low frequency categories as 'other'
x['specific column']=['i'if x=='i'else 'other']
print['specific column'].value_counts().sort_values(ascending=False)
#create a list of features to dummy
todummy_list= ['n']
#function to dummy all the categorical variables used for modeling
def dummy_df(df,todummy_list):
    for x in todummy_list:
        dummies=pd.get_dummies(df[x,prefix=x],dummy_na=False)
        df=df.drop(x,1)
        df=pd.concat([df,dummies],axis=1)
    return df
x=dummy_df(x,to dummy_list)
print(x.head())

#Handling Missing Data
#how much of your data is missing
x.isnull().sum().sort_values(ascending=false).head()
#inpute missing values using Imputer in sklearn
from sklearn.preprocessing import Inputer
imp= Inputer(missing_value='NaN',strategy='Medium', axis=0)
x=pd.DataFrame(data=imp.transform(x), columns=x.columns)
#check again to see if u still have missing data
x.isnull().sum().sort_values(ascending+false)head()

#Outlier Detection(Tukey IQR)
def find_outliers_tukey(x):
    q1=np.percentile(x,25)
    q3=np.percentile(x,75)
    iqr=q3-q1
    floor=q1-1.5*iqr
    ceiling=q3+1.5*iqr
    outlier_indices=list(x.index[(x<floor)|(x>ceiling)])
    outlier_values]=list(x[outlier_indices])
    return outlier_indices,outlier_values

tukey_indices,tukey_values=find_outliers_tukey(x[''])
print(np.sort(tukey_values))

#Outlier Detection(Kernel Density Estimation)
from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate

def find_outliers_kde(x):
    x_scaled=scale(list(map(float,x)))
    kde= KDEUnivariate(x_scaled)
    kde.fit(bw="scott",fit=True)
    pred=kde.evaluate(x_scaled)

    n=sum(pred<0.05)
    outlier_index=np.asarray(pred).argsort()[:n]
    outlier_value=np.asarray(x)[outlier_ind]
    
    return outlier_ind,outlier_value

kde_indices,kde_values=find_outliers_kde(x[''])
print(np.sort(kde_values))

#Distribution of Features

#use pyplot in matplotlib to plot histogram
import matplotlib.pyplot as plt 
def plot_histogram(x):
        plt.hist(x, color='',alpha=)
        plt.title("Hostogram of '{var_name}'".format(var_name=x.name))
        plt.xlabel("value")
        plt.ylabel("Frequency")
        plt.show()
plot_histogram(x[''])

#plot histogram to show distribution of features by Dependant Variables(DV) 
def plot_histogram_dv(x,y):
        plt.hist(list(x[y==0]),alpha=''label='DV=0')
        plt.hist(list(x[y==1]),alpha=''label='DV=1')
        plt.title("Histogram of '{var_name}' by DV category".format(var_name=x.name))
        plt.xlabel("Value")
        plt.ylabe("Frequency")
        plt.legend(loc='upper right')
        plt.show()
plot_histogram_dv(x[''],y)

# Feature Engeneering
# a. Interactions of Features(increasing dimensionality)
#use polynomial features in sklearn.preprocessing to create two-way interactions for all features

from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
        #Get feature names
        combos=list(combinations(list(df.columns),2))
        colnames=list(df.columns)+['_'.join(x) for x in combos]

        #Find interactions 
        poly=PolynomialFeatures(interactions_only=True, include_bias=False)
        df=poly.fit_transform(df)
        df=pd.DataFrame(df)
        df.colums=colnames

        #remove interactions terms with all 0 values
        noint_indicies=[i for i, x in enumerate(list((df==0).all())) if x]
        df=df.drop(df.columns[noint_indicies],axis=1)

        return df

x=add_interactions(x)
print(x)

#PCA
#Use PCA from sklearn.decomposition to find principal components
from sklearn.decomposition import PCA

pca=PCA(n_components=10)
x_pca=pd.DataFrame(pca.fit_transform(x))

print(x.pca)

#Use feature selection to select the most important features 
#RFE
import sklearn.feature_selection

select=sklearn.feature_selection.SelectKBest(k=20)
selected_features=select.fit(x_train,y_train)
indices_selected=selected_features.get_support(indices=True)
colnames_selected=[x.columns[i] for i in indices_selected]

x_train_selected=x_train[colnames_selected]
x_test_selected=x_test[colnames_selected]
colnames_selected