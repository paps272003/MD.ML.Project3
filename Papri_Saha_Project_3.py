# Import packages and librairies
##https://pandas.pydata.org/pandas-docs/stable/tutorials.html
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
import pandas as pd

#Read the csv file. Double quote is used to remove the uniecode character
df_raw= pd.read_csv('C:\\Users\\a481264\\Desktop\\POCPython\\adult.data')

#Check the num of rows and coulmns
df_raw.shape

#Renaming the column names since the first row is getting captured as column name by default

df_raw.columns=['age','workclass','fnlwgt','education','education-num','marital-status'
                ,'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week'
                ,'native-country','label']

#Check for all columns both numeric and non-numeric
#Describing all columns of a DataFrame regardless of data type.
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html
stat=df_raw.describe(include='all')


###Data Exploration & Visualization

#Find the unique values for the column marital status
df_raw['marital-status'].unique()
#For each unique values in the column what are the counts for each of them
df_raw['marital-status'].value_counts()

#Find the unique values for the column workclass
df_raw['workclass'].value_counts()

#Find the unique values for the column occupation
df_raw['occupation'].value_counts()

#Check for the nulls or missing values in the dataset

df_raw.isnull().values.any()

#label value count
df_raw['label'].value_counts()

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

sns.countplot(x='label',data=df_raw, palette='hls')
plt.show()
plt.savefig('count_plot')

###Let's get a sense of the numbers across the two classes

df_raw.groupby('label').mean()

df_raw.groupby('occupation').mean()

df_raw.groupby('relationship').mean()

df_raw.groupby('education-num').mean()

#Create a copy of the raw dataframe
df=df_raw.copy()
df.isnull().sum()



#Drop columns that are not needed and might not be contributing to prediction
df['education'].value_counts()
df['education-num'].value_counts()
df.drop("education", axis=1, inplace=True)
df.drop("marital-status", axis=1, inplace=True)
df.drop("relationship", axis=1, inplace=True)

#Data Visualization
#https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/
#https://www.analyticsvidhya.com/blog/2014/07/statistics/
#https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/
#https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
df.boxplot(column='hours-per-week')

df['hours-per-week'].hist(bins=50)

df.boxplot(column='hours-per-week', by = 'education-num',figsize=(12,10))

#Separating the X matrix of features and Y target vector 
X = df.iloc[:, :-1].values
y = df.iloc[:, 11].values

#Converting the numpy array to dataframe
#X_temp=pd.DataFrame(data=X[0:,0:],columns=['age','workclass','fnlwgt','education-num','occupation','race','sex','capital-gain','capital-loss','hours-per-week','native-country'])
#y_temp=pd.DataFrame(data=y,columns=['label'])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
X[:, 10] = labelencoder_X.fit_transform(X[:, 10])

#Post label encoder apply the one hot encoding by passing the indices of the columns which are categorical
onehotencoder = OneHotEncoder(categorical_features = [1,4,5,6,10])
X = onehotencoder.fit_transform(X)
#Convert the sparse matrix into array
X=X.toarray()

# Encoding the Dependent Variable.We shouldnot apply the one hot encoding as it is the target or dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results for logistic regression
y_pred = classifier.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

#Cross Validation
'''Cross validation attempts to avoid overfitting while still producing a prediction for each observation dataset. 
We are using 10-fold Cross-Validation to train our Logistic Regression model.'''

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

'''The average accuracy remains very close to the Logistic Regression model accuracy; 
hence, we can conclude that our model generalizes well.'''

# Making the Confusion Matrix of Logistic regression
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Compute precision, recall, F-measure and support
'''The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, 
where an F-beta score reaches its best value at 1 and worst score at 0'''

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

'''Interpretation: Of the entire test set, 82% of the census were given the salary which is less than 50k. 
Of the entire test set, 83% of the census got an annual income which is less than 50k.'''

#ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
'''The dotted line represents the ROC curve of a purely random classifier; 
a good classifier stays as far away from that line as possible (toward the top-left corner)'''

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier_knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred)
print(cm_knn)

# Fitting RF to the Training set
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

classifier_RF = RandomForestClassifier(criterion='entropy',max_depth=7,
min_samples_split=10, random_state=99)

classifier_RF.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RF = classifier_RF.predict(X_test)

#Testing RF accuracy on test set 
accuracy_score(y_test, y_pred_RF)

#Alternative Testing RF accuracy on test set 
classifier_RF.score(X_test, y_test)

#Splitting the dataset into test and train using K-fold model
from sklearn.cross_validation import KFold
crossvalidation = KFold(n=df.shape[0], n_folds=10, shuffle=True,random_state=1)

#Printing the training set and test set through cross validation
#http://pythonforengineers.com/cross-validation-and-model-selection/
'''for train_set,test_set in crossvalidation:
    print(train_set, test_set)'''

classifier_RF_CV = RandomForestClassifier(criterion='entropy',max_depth=7,min_samples_split=20, random_state=99)
classifier_RF_CV.fit(X,y)
from sklearn.cross_validation import cross_val_score
score = np.mean(cross_val_score(classifier_RF_CV, X, y, scoring='accuracy',cv=crossvalidation, n_jobs=1))
print("Accuracy of Random Forests is: " , score)

RFE=classifier_RF_CV.feature_importances_


