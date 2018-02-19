from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
rom sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a dataframe with the four feature variables
bact_f = pd.read_table('proteobacteria-f_fitch-UPGMA.txt')
bact_g = pd.read_table('proteobacteria-g_fitch-UPGMA.txt')
fun = pd.read_table('fungi_fitch-UPGMA.txt')
euk = pd.read_table('eukaryota_fitch-UPGMA.txt')
archaea = pd.read_table('archaea_fitch-UPGMA.txt')

# merging the training set
frames = [archaea, fun, euk]
train = pd.concat(frames)

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
print(precision_score(y_test, preds))


# Create confusion matrix
#print(accuracy_score(x, test['preds']))
#print(pd.crosstab(y_test, test['preds']))
#print(f1_score(y_test, test['preds']))

plt.plot(fpr, tpr, color='darkorange')
plt.savefig("out.svg")
print(roc_auc_score(y_test, predictions))

# precision-recall plot dodelai i perenesi otsuda v 10-fold!!!
precision, recall, _ = precision_recall_curve(y_test, predictions)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
