import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df=pd.read_csv('loan_data.csv')
'''
print(df.head())
print(df.tail())
print(df.describe())
print(df.columns)
print(df.dtypes)
print(df.isna().sum())


#EDA
#A complete scatter plot between all the features
#adjust the figure size
plt.figure(figsize=(20, 20))
#make a pairplot for whole dataset
sns.pairplot(data=df, hue='not.fully.paid')
#adjust a tight and well gaped layout
plt.tight_layout()
plt.show()

#1.Univariant data analysis
#feature distribution
df.hist(bins=60, figsize=(20, 25))
plt.suptitle('Feature Distribution', x=0.5, y=1.02, ha='center', fontsize='large')
plt.show()

#boxplot for all the features
df.boxplot(figsize=(10,8))
plt.suptitle('BoxPlot of Feature', x=0.5, y=1.02, ha='center', fontsize='large')
plt.xticks(rotation=90)
plt.show()


#2. Bivariant data analysis
#adjust size of the plot
plt.figure(figsize=(15,8))
#number of fico w.r.t credit policy
df[df['credit.policy']==0]['fico'].hist(bins=30, color='red', label='Credit Policy=0')
df[df['credit.policy']==1]['fico'].hist(bins=30, color='blue', label='Credit Policy=1')
plt.xlabel('FICO')
plt.show()


#adjust size of the plot
plt.figure(figsize=(15, 8))
#number of fico w.r.t not fully paid
df[df['not.fully.paid']==0]['fico'].hist(bins=30, alpha=0.5, label='Not fully paid=0')
df[df['not.fully.paid']==1]['fico'].hist(bins=30, alpha=0.5, label='Not fully paid=1')
plt.xlabel('FICO')
plt.show()


#adjust size of the plot
plt.figure(figsize=(15, 8))
#check the density of the number of days the borrower has had a credit line
sns.distplot(x=df['days.with.cr.line'], kde=True, hist=True)
plt.xlabel('The number of days the borrower has had a credit line')
plt.show()


#adjust size of the plot
plt.figure(figsize=(15, 8))
#create a boxplot of different purpose w.r.t fico
sns.boxplot(x='purpose', y='fico', data=df)
#fit the plot
plt.show()


#adjust size of the plot
plt.figure(figsize=(15, 8))
#countplot for different purpose
sns.countplot(data=df, x='purpose', hue='credit.policy')
plt.xlabel('Purpose')
plt.show()


#adjust size of the plot
plt.figure(figsize=(15, 8))
#countplot for different purpose
sns.countplot(data=df, x='purpose', hue='not.fully.paid')
plt.xlabel('Purpose')
plt.show()


#adjust size of the plot
plt.figure(figsize=(8,7))
#scatter plot for fico vs interest rate(in decimals)
sns.jointplot(data=df, x='fico', y='int.rate', hue='not.fully.paid')
plt.show()


#adjust size of the plot
plt.figure(figsize=(15, 8))
sns.lmplot(data=df, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid')
plt.show()


#adjust size of the plot
plt.figure(figsize=(15, 8))
#check the density of the amount of debt divided by annual income
sns.distplot(x=df['dti'], kde=True, hist=True)
plt.xlabel('The amount of debt divided by annual income')
plt.show()


#correlation between features of dataset
print(df.corr())


#adjust size for the heatmap
plt.figure(figsize=(12,10))
#create a heatmap for all features(expect purpose)
sns.heatmap(df.corr(), annot=True, linecolor='white', linewidths=0.2)
plt.show()
'''

#Hot Encoding(Purpose column has object data type and we know that machine learning model doesn't work for categorical or object data type. So, we need to convert object type into numerical form and after that we will able to train and test our machine learning model)
purpose_col=['purpose']
df_final=pd.get_dummies(data=df, columns=purpose_col, drop_first=True)
print(df_final.head())
print(df_final.columns)


#Machine learning model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#scaling the dataset
scaler=StandardScaler()
scaler.fit(df_final.drop('not.fully.paid', axis=1))

#transform the data in scaler instance
scaler_feature=scaler.transform(df_final.drop('not.fully.paid', axis=1))

#make a final scaled dataframe which is going to use for our model
columns=['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti',
       'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
       'inq.last.6mths', 'delinq.2yrs', 'pub.rec',
       'purpose_credit_card', 'purpose_debt_consolidation',
       'purpose_educational', 'purpose_home_improvement',
       'purpose_major_purchase', 'purpose_small_business']
df_scale=pd.DataFrame(scaler_feature, columns=columns)
print(df_scale.head())

#split the training features and target variable
X=df_scale
y=df_final['not.fully.paid']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)
#make an instance for all ml algorithms
lr=LogisticRegression()
dtree=DecisionTreeClassifier()
rfc=RandomForestClassifier(n_estimators=200)
svm=SVC(probability=True)
knn=KNeighborsClassifier(n_neighbors=5)

#make a list for all ml instance
ml_algo=[lr, dtree, rfc, svm, knn]
#find the MSE of all the algorithms and selected top three algo, with least MSE.
for i in ml_algo:
    i.fit(X_train, y_train)
    pred=i.predict(X_test)
    print(i, 'MSE:{:4F}'.format(mean_squared_error(y_test, pred)), '\n')


#Logistic Regression
#fitting the data and make a model with logistic regression ml algo.
lr.fit(X_train, y_train)
#predict the test data
lr_pred=lr.predict(X_test)

#check confusion matrix, classification report and accuracy score of prediction value
cm=confusion_matrix(y_test, lr_pred)
print(cm, '\n')
print(classification_report(y_test, lr_pred), '\n')
print('Accuracy Score:{:.2f}%'.format(accuracy_score(y_test, lr_pred)*100))

plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm,annot=True,fmt='d', cmap='coolwarm', square=True,xticklabels=['0', '1'],yticklabels=['0', '1'])
plt.xlabel('Predicted Values') # x label of the confusion matrix
plt.ylabel('True Values') # y label of the confusion matrix
plt.title('Confusion Matrix for Logistic Regression') # title of the confusion matrix

# create a ROC Curve and AUC plot to assess the overall diagnostic performance of a test and to compare the performance of
# two or more diagnostic tests.

lr_pred_prob = lr.predict_proba(X_test)[:][:,1]

lr_actual_predict = pd.concat([pd.DataFrame(np.array(y_test),columns=['y actual']),
                               pd.DataFrame(lr_pred_prob,columns=['y pred prob'])],axis=1)
lr_actual_predict.index = y_test.index

fpr, tpr, tr = roc_curve(lr_actual_predict['y actual'],lr_actual_predict['y pred prob'])
auc = roc_auc_score(lr_actual_predict['y actual'],lr_actual_predict['y pred prob'])

plt.plot(fpr,tpr, label='AUC=%.4f'%auc)
plt.plot(fpr,fpr,linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Reg.')
plt.legend()

#Random Forest
# fitting data and make a model from random forest ml algo
rfc.fit(X_train,y_train)

# predict the test data
rfc_pred = rfc.predict(X_test)

# check confusion matrix, classification report and accuracy score of predicted values
cm = confusion_matrix(y_test,rfc_pred)
print(cm,'\n')
print(classification_report(y_test,rfc_pred),'\n')
print("Accuracy Score: {:.2f}%".format(accuracy_score(y_test,rfc_pred)*100))

plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm,annot=True,fmt='d', cmap='coolwarm', square=True,xticklabels=['0', '1'],yticklabels=['0', '1'])
plt.xlabel('Predicted Values') # x label of the confusion matrix
plt.ylabel('True Values') # y label of the confusion matrix
plt.title('Confusion Matrix for Random Forest') # title of the confusion matrix

# ROC and AUC for the random forest model
rfc_pred_prob = rfc.predict_proba(X_test)[:][:,1]

rfc_actual_predict = pd.concat([pd.DataFrame(np.array(y_test),columns=['y actual']),
                               pd.DataFrame(rfc_pred_prob,columns=['y pred prob'])],axis=1)
rfc_actual_predict.index = y_test.index

fpr, tpr, tr = roc_curve(rfc_actual_predict['y actual'],rfc_actual_predict['y pred prob'])
auc = roc_auc_score(rfc_actual_predict['y actual'],rfc_actual_predict['y pred prob'])

plt.plot(fpr,tpr, label='AUC=%.4f'%auc)
plt.plot(fpr,fpr,linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend()

#SVM
# fitting data and make a model from svm ml algo.
svm.fit(X_train,y_train)

# predict the test data
svm_pred = svm.predict(X_test)

# check confusion matrix, classification report and accuracy score of predicted values
cm = confusion_matrix(y_test,svm_pred)
print(cm,'\n')
print(classification_report(y_test,svm_pred),'\n')
print("Accuracy Score: {:.2f}%".format(accuracy_score(y_test,svm_pred)*100))

# adjust the size of the confusion matrix
plt.figure(figsize=(6, 6))
sns.set(font_scale=1.2)

# create a heatmap which show the confusion matrix of the SVM model
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', square=True,xticklabels=['0', '1'],yticklabels=['0', '1'])

plt.xlabel('Predicted Values') # x label of the confusion matrix
plt.ylabel('True Values') # y label of the confusion matrix
plt.title('Confusion Matrix for SVM') # title of the confusion matrix

# ROC and AUC for the SVM model
svm_pred_prob = svm.predict_proba(X_test)[:][:,1]

svm_actual_predict = pd.concat([pd.DataFrame(np.array(y_test),columns=['y actual']),
                               pd.DataFrame(svm_pred_prob,columns=['y pred prob'])],axis=1)
svm_actual_predict.index = y_test.index

fpr, tpr, tr = roc_curve(svm_actual_predict['y actual'],svm_actual_predict['y pred prob'])
auc = roc_auc_score(svm_actual_predict['y actual'],svm_actual_predict['y pred prob'])

plt.plot(fpr,tpr, label='AUC=%.4f'%auc)
plt.plot(fpr,fpr,linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM')
plt.legend()

#Hyperparameter Tuning
# hyperparameter tuning for Logistic Regression Model
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'max_iter': [100, 200, 300]
}

# hyperparameter tuning for Random Forest Model
param_grid_rfc = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# hyperparameter tuning for SVM Model
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma':[0.001, 0.01, 0.1, 1]
}

#Logistic Regression
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train,y_train)

print('Best Hyperparameters for Logistic Regression Model:',grid_search_lr.best_params_)

#Get the best model
best_lr_model = grid_search_lr.best_estimator_

# Make predictions with the best model
y_lr_pred = best_lr_model.predict(X_test)

#confusion matrix, classification report and accuracy score
print(confusion_matrix(y_test,y_lr_pred),'\n')
print(classification_report(y_test,y_lr_pred),'\n')
print('Accuracy score of best model: {}%'.format(accuracy_score(y_test,y_lr_pred)*100))

#random Forest
grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rfc, cv=5, scoring='accuracy')
grid_search_rfc.fit(X_train,y_train)

print('Best Hyperparameters for Random Forest Model:',grid_search_rfc.best_params_)

# Get the best model
best_rfc_model = grid_search_rfc.best_estimator_

# Make predictions with the best model
y_rfc_pred = best_rfc_model.predict(X_test)

#confusion matrix, classification report and accuracy score
print(confusion_matrix(y_test,y_rfc_pred),'\n')
print(classification_report(y_test,y_rfc_pred),'\n')
print('Accuracy score of best model: {}%'.format(accuracy_score(y_test,y_rfc_pred)*100))

#Support Vector Machine (SVM)
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train,y_train)

print('Best Hyperparameters for Random Forest Model:',grid_search_svm.best_params_)

# Get the best model
best_svm_model = grid_search_svm.best_estimator_

# Make predictions with the best model
y_svm_pred = best_svm_model.predict(X_test)

#confusion matrix, classification report and accuracy score
print(confusion_matrix(y_test,y_svm_pred),'\n')
print(classification_report(y_test,y_svm_pred),'\n')
print('Accuracy score of best model: {}%'.format(accuracy_score(y_test,y_svm_pred)*100))

'''
Comparison of Accuracy Score Before and After Hyperparameter Tuning
        Model                  Accuracy Score (Before Hyper. Tuning)       AccuracyScore (After Hyper. Tuning)
   1. Logistic Reg.            83.82%                                      83.71%
   2. Random Forest            83.65%                                      83.78%
   3. SVM                      83.82%                                      83.78%
So, As we able to see that, hyperparameter tuning doesn't effect well in our models like:- except Random Forest, other models accuracy decrease (but, very less). Random Forest accuracy increase slightly (0.13%) which is not enough but, they do not reduce the accuracy, that is good for us.

#Conclusion
Now, if we have to conclude that which model is good for our dataset or problem then, it is difficult to conclude because all three models work well. But, we also have to focus on precision, recall and f1-score because if we observe classification report and confusion matrix then, we got error of type I and type II.

Hence, we conclude that our models is not much good for the prediction. We have to do some more work on our models to make best predictions for our future data.
'''