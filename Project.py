import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#importing our dataset
dataset =pd.read_csv('Tumor Cancer Prediction_Data.csv')

#print(dataset.info())

#return the size of dataset
print(dataset.shape)

#return all the columns with null values
#print(dataset.isna().sum())

#preprocessing with missing value
data = dataset.dropna()

#count of M and B cells
df = dataset['diagnosis'].value_counts()
print(df)

x=dataset.iloc[:, 1:30].values
y=dataset.iloc[:, 31].values

#encoding categorical data value
labelencoder_Y =LabelEncoder()
y=labelencoder_Y.fit_transform(y)

#splitting dataset into training&test
x_train ,x_test , y_train , y_test=train_test_split(x,y,test_size=0.3 , random_state =109)

#feature scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#models algorithms
def models (x_train ,y_train):
    # using logistic regrition
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    print('LogisticRegression : ', clf.score(x_test, y_test))

    # using kneighbour clacifier
    kn = neighbors.KNeighborsClassifier()
    kn.fit(x_train, y_train)
    print('kneighbour clacifier : ', kn.score(x_test, y_test))

    # using svm model
    svc = svm.SVC(kernel='linear')
    svc.fit(x_train, y_train)
    print('svm linear : ', svc.score(x_test, y_test))


    #using decision tree
    tree = DecisionTreeClassifier(criterion="entropy")
    tree.fit(x_train,y_train)
    print('Decision tree', tree.score(x_test, y_test))

    return clf, kn, svc,tree
#testing the models/result
model = models(x_train,y_train)
for i in range(len(model)):
    print("model" , i+1)
    print(classification_report(y_test , model[i].predict(x_test)))
    print('Accuracy : ', accuracy_score(y_test , model[i].predict(x_test)))
    print()

for i in range(len(model)):
    pred = model[i].predict(x_test)
    print('predicted values ' , i+1)
    print(pred)
    print('actual values ' , i+1)
    print(y_test)
    print()

