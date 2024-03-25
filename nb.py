# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.inspection import permutation_importance

# Directory
curDir = os.getcwd()

# Reading earthquake file
df = pd.read_csv('cleaned_earthquake.csv')

# Choosing columns for training and prediction
features = df[['magnitude', 'mmi', 'tsunami', 'dmin', 'gap', 'depth', 'nst', 'latitude', 'longitude']]
output = df['alert']
output, mapping = pd.factorize(output)

# Pre-model process
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    output,
                                                    test_size=0.2,
                                                    random_state=41)

# Modelling
nbmodel = GaussianNB()
nbmodel.fit(x_train, y_train)
y_pred = nbmodel.predict(x_test)

# To see how prediction can be reliable
accuracy = accuracy_score(y_pred, y_test)
f1score = round(f1_score(y_test, y_pred, average='weighted'), 2)
precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
print('Accuracy:', accuracy)
print('f1 score:', f1score)
print('Precision:', precision)

# Creating file to store trained model
model_path = os.path.join(curDir, 'nbm.pickle')
with open(model_path, 'wb') as nb_pickle:
    pickle.dump(nbmodel, nb_pickle)
    nb_pickle.close()

# Creating file to store output    
model_path2 = os.path.join(curDir, 'outputnb.pickle')
with open(model_path2, 'wb') as outputnb_pickle:
    pickle.dump(mapping, outputnb_pickle)
    outputnb_pickle.close()

# Finding out the each column's importance for prediction
imp = permutation_importance(nbmodel, x_test, y_test)
fig, ax = plt.subplots() 
ax = sns.barplot(x=imp.importances_mean, y=features.columns) 
plt.title('Which features are the most important for species prediction?') 
plt.xlabel('Importance of Features') 
plt.ylabel('Features of Earthquake') 
plt.tight_layout() 
fig.savefig('features_importance.png')