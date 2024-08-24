import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np


# import dataset and add headers
headers = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("pima-indians-diabetes.data.csv",names=headers)
print(data)
print(data.groupby('class').size())

# Replace missing values with mean
print((data[['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']]== 0).sum())
data_cp = data.copy(deep=True)
data_cp[['preg','plas','pres','skin','test','mass']] = data_cp[['preg','plas','pres','skin','test','mass']].replace(0,np.NaN)
print(data_cp.head(10))
print(data_cp.isnull().sum())
data_missing = data_cp.dropna()
dataset = data_cp.fillna(data_cp.mean())
nrow, ncol = dataset.shape
predictors = dataset.iloc[:,:ncol-1]
target = dataset.iloc[:,-1]

# Split into train and test
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size= 0.3, random_state = 42 )

# Decision tree classifier where max depth is increased in steps of 2 for every iteration
Decision_tree = []
cols = ['Max Depth', 'Accuracy']

Max_depth_threshold = 21
for i in range (2,Max_depth_threshold,2):
    classifier = DecisionTreeClassifier(max_depth=i,random_state=42)
    classifier = classifier.fit(pred_train,tar_train)
    predictions = classifier.predict(pred_test)
    Decision_tree.append([i,accuracy_score(tar_test,predictions)])

Decision_tree_df = pd.DataFrame(Decision_tree, columns=cols)
print(Decision_tree_df)
print("")
Max_Depth,Max_Accuracy = Decision_tree_df.idxmax()
print("Best value of max depth")
print(Decision_tree_df.iloc[Max_Accuracy])
print("")

# Neural Network (MLP)
Min_learning_Rate = 0.001
Max_Learning_Rate = 0.011
Neural_network = []
cols_2 = ['Learning Rate', 'Accuracy']
while Min_learning_Rate <= Max_Learning_Rate:
    classifier_2 = MLPClassifier(learning_rate_init= Min_learning_Rate, random_state=42)
    classifier_2 = classifier_2.fit(pred_train, np.ravel(tar_train, order='C'))
    predictions_2 = classifier_2.predict(pred_test)
    Neural_network.append([Min_learning_Rate,accuracy_score(tar_test,predictions_2)])
    Min_learning_Rate += 0.001

Neural_network_df = pd.DataFrame(Neural_network, columns=cols_2)
print(Neural_network_df)
print("")

Best_learning_rate, Best_Accuracy = Neural_network_df.idxmax()
print("Best learning rate for Neural Network")
print(Neural_network_df.iloc[Best_Accuracy])
print("")


