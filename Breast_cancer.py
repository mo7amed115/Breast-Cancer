# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import  confusion_matrix 

# Importing the data .
data = pd.read_csv('data/breast_cancer.csv')

# Test missing data .
data.isnull().sum()

# Features Selection using correlation method.
corr = data.corr()

# Visualise the full data
fig , ax = plt.subplots(figsize = (40 ,40 ))
ax = sns.heatmap(corr ,
                 annot = True ,
                 linewidths=0.5 ,
                 fmt= '0.2f' ,
                 cmap = 'YlGnBu')

# drop the columns innessesary .
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
data = data[selected_columns]

# Visualise the New data
corr2 = data.corr()
fig , ax2 = plt.subplots(figsize = (40 , 40 ))
ax2 = sns.heatmap(corr2 ,
                  annot = True ,
                  linewidths=0.5 ,
                  fmt= '0.2f' ,
                  cmap = 'YlGnBu')

# Spliting Dataset to Features and Target
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# Spliting Dataset to Training and Testing
X_train , X_test , y_train , y_test = train_test_split(X ,
                                                        y ,
                                                        test_size=0.3 ,
                                                        shuffle = True ,
                                                        random_state = 0)

# Scalling the columns data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Training Data and Predict the Results .
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {'Logestic_Regression' : LogisticRegression() ,
          'KNN' : KNeighborsClassifier() ,
          'Random_Forest_Classifier' : RandomForestClassifier() ,
          'SVC' : SVC()
          }
def fit_and_score(models , X_train , X_test , y_train , y_test) :
    model_scores = {}
    model_confusion = {}
    for name , model in models.items() :
        # fitting the data :
        model.fit(X_train , y_train)
        model_scores[name] = model.score(X_test , y_test)
        y_predict = model.predict(X_test)
        model_confusion[name] = confusion_matrix(y_test , y_predict)
    return model_scores , model_confusion
    
fit_and_score( models = models ,
               X_train = X_train , X_test = X_test ,
               y_train = y_train , y_test = y_test
             )
