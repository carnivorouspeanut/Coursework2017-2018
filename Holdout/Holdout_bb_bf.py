from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a dataframe
bact_f = pd.read_table('proteobacteria-f_fitch-UPGMA.txt')
bact_g = pd.read_table('proteobacteria-g_fitch-UPGMA.txt')

train = bact_g
test = bact_f

# Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75.
#df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# Create two new dataframes, one with the training rows, one with the test rows
#train, test = df[df['is_train']==True], df[df['is_train']==False]

# Create a list of the feature column's names
features = train.columns[1:-1]

X_train = train[features].as_matrix()
X_test = test[features].as_matrix()

y_train = train['division'] == 'fitch'
y_test = test['division'] == 'fitch'


# Create a random forest Classifier
clf = RandomForestClassifier(n_estimators = 1000)

# Train the Classifier to take the training features and learn how they relate to the training y (the species)
classifier = clf.fit(X_train, y_train)
predictions = pd.DataFrame(data = classifier.predict_proba(X_test))
predictions = predictions[1]


preds = clf.predict(X_test)
test['preds'] = preds
fpr, tpr, thresholds = roc_curve(y_test, predictions)
# Calculate precision
print(precision_score(y_test, preds))


# Create confusion matrix
#print(accuracy_score(x, test['preds']))
#print(pd.crosstab(y_test, test['preds']))
#print(f1_score(y_test, test['preds']))

plt.plot(fpr, tpr, color='darkorange')
plt.savefig("out.svg")
print(roc_auc_score(y_test, predictions))
