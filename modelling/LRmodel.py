import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, accuracy_score, f1_score, precision_score


curDir = os.getcwd()

df = pd.read_csv('cleaned_earthquake.csv')

features = df[['magnitude', 'mmi', 'tsunami', 'dmin', 'gap', 'depth', 'nst', 'latitude', 'longitude']]
output = df['alert']

output, mapping = pd.factorize(output)

x_train, x_test, y_train, y_test = train_test_split(features,
                                                    output,
                                                    test_size=0.2,
                                                    random_state=41)

reg_model = LinearRegression()
reg_model.fit(x_train, y_train)
y_pred = reg_model.predict(x_test)

y_pred_classes = pd.cut(y_pred, bins=3, labels=False)

r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)
accuracy = accuracy_score(y_test, y_pred_classes)
print('Accuracy:', accuracy)

f1score = f1_score(y_test, y_pred_classes, average='weighted')
print('F1 Score:', f1score)

precision = precision_score(y_test, y_pred_classes, average='weighted')
print('Precision:', precision)

model_path = os.path.join(curDir, 'linearregression.pickle')
with open(model_path, 'wb') as reg_pickle:
    pickle.dump(reg_model, reg_pickle)
 
model_path2 = os.path.join(curDir, 'output_linearregression.pickle')
with open(model_path2, 'wb') as output_pickle:
    pickle.dump(mapping, output_pickle)

imp = permutation_importance(reg_model, x_test, y_test)
fig, ax = plt.subplots() 
ax = sns.barplot(x=imp.importances_mean, y=features.columns) 
plt.title('Importance of Features for Prediction') 
plt.xlabel('Importance') 
plt.ylabel('Features') 
plt.tight_layout() 
fig.savefig('LRfeatures_importance.png')
