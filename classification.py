import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(a,b,c,d,e,f,g,h,i,j,k,l,m,n):
    fig, ax = plt.subplots(figsize=(100,100))
    ax.plot(a,b, color='orange', label='Logisitc Regression AUC: %.2f' % auc_lr)
    ax.plot(c,d, color='red', label='SVM AUC: %.2f' % auc_svm)
    ax.plot(e,f, color='yellow', label='Kernel SVM AUC: %.2f' % auc_ksvm)
    ax.plot(g,h, color='magenta', label='Naive Bayes AUC: %.2f' % auc_nb)
    ax.plot(i,j, color='green', label='Decision Tree AUC: %.2f' % auc_dt)
    ax.plot(k,l, color='black', label='Random Forest AUC: %.2f' % auc_rf)
    ax.plot(m,n, color='cyan', label='KNN AUC: %.2f' % auc_knn)
    ax.set_xlabel('False Positive Rate', fontsize=13.5, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13.5, fontweight='bold')
    plt.xticks(fontsize=13.5)
    plt.yticks(fontsize=13.5)
    plt.title('ROC Curves',fontweight='bold',fontsize=15.5)
    plt.legend(loc=4,prop={'size':16})
    plt.show()
# Importing the dataset
dataset = pd.read_csv('newfeatures.csv')
X = dataset.iloc[:, 0:19].values
y = dataset.iloc[:, 19].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)
y_pred_lr = classifier_lr.predict(X_test)

from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0,probability=True)
classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)

from sklearn.svm import SVC
classifier_ksvm = SVC(kernel = 'rbf', random_state = 0,probability=True)
classifier_ksvm.fit(X_train, y_train)
y_pred_ksvm = classifier_ksvm.predict(X_test)

from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_pred_nb = classifier_nb.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)
y_pred_dt = classifier_dt.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)
y_pred_knn = classifier_knn.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Logistic Regression")
print(classification_report(y_test,y_pred_lr))

cm_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM")
print(classification_report(y_test,y_pred_svm))

cm_ksvm = confusion_matrix(y_test, y_pred_ksvm)
print("Kernel SVM")
print(classification_report(y_test,y_pred_ksvm))

cm_nb = confusion_matrix(y_test, y_pred_nb)
print("Naive Bayes")
print(classification_report(y_test,y_pred_nb))

cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Decision Tree")
print(classification_report(y_test,y_pred_dt))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest")
print(classification_report(y_test,y_pred_rf))

cm_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN")
print(classification_report(y_test,y_pred_knn))


probs_lr = classifier_lr.predict_proba(X_test)
probs_lr = probs_lr[:, 1]
auc_lr = roc_auc_score(y_test, probs_lr)
print('Logisitc AUC: %.2f' % auc_lr)
fpr_lr, tpr_lr, thresholds = roc_curve(y_test, probs_lr)

probs_svm = classifier_svm.predict_proba(X_test)
probs_svm = probs_svm[:, 1]
auc_svm = roc_auc_score(y_test, probs_lr)
print('SVM AUC: %.2f' % auc_svm)
fpr_svm, tpr_svm, thresholds = roc_curve(y_test, probs_svm)

probs_ksvm = classifier_ksvm.predict_proba(X_test)
probs_ksvm = probs_ksvm[:, 1]
auc_ksvm = roc_auc_score(y_test, probs_ksvm)
print('Kernel SVM AUC: %.2f' % auc_ksvm)
fpr_ksvm, tpr_ksvm, thresholds = roc_curve(y_test, probs_ksvm)

probs_nb = classifier_nb.predict_proba(X_test)
probs_nb = probs_nb[:, 1]
auc_nb = roc_auc_score(y_test, probs_nb)
print('Naive Bayes AUC: %.2f' % auc_nb)
fpr_nb, tpr_nb, thresholds = roc_curve(y_test, probs_nb)

probs_dt = classifier_dt.predict_proba(X_test)
probs_dt = probs_dt[:, 1]
auc_dt = roc_auc_score(y_test, probs_dt)
print('Decision Tree AUC: %.2f' % auc_dt)
fpr_dt, tpr_dt, thresholds = roc_curve(y_test, probs_dt)

probs_rf = classifier_rf.predict_proba(X_test)
probs_rf = probs_rf[:, 1]
auc_rf = roc_auc_score(y_test, probs_rf)
print('Random Forest AUC: %.2f' % auc_rf)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, probs_rf)

probs_knn = classifier_knn.predict_proba(X_test)
probs_knn = probs_knn[:, 1]
auc_knn = roc_auc_score(y_test, probs_knn)
print('KNN AUC: %.2f' % auc_knn)
fpr_knn, tpr_knn, thresholds = roc_curve(y_test, probs_knn)


accuracies={'CNN':66,'Logistic Regression':69.7,'SVM':68.75,'Kernel_SVM':76.04,'Naive Bayes':58.33,"Decision Tree":73.958,"Random Forest":83.33,"KNN":76.04}
fig, ax = plt.subplots(figsize=(100,100))
ax.bar(*zip(*sorted(accuracies.items())),color='b')
ax.set_xlabel('Classifiers', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
plt.xticks(fontsize=13.5)
plt.yticks(fontsize=13.5)
plt.title('Accuracy Graph',fontsize=15.5,weight='bold')
plt.show()
plot_roc_curve(fpr_lr, tpr_lr,fpr_svm, tpr_svm,fpr_ksvm, tpr_ksvm,fpr_nb, tpr_nb,fpr_dt, tpr_dt,fpr_rf, tpr_rf,fpr_knn, tpr_knn)
