# installl sklearn
pip install sklearn
pip install numpy as np
pip install pydot
pip install graphviz

# import the training datasets
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

# training the data sets
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data 
test_target = iris.target[test_idx] 
test_data = iris.data[test_idx]# tarin the model
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# expected output
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
                       

print(test_target)

#output
[0 1 2]

# test for prediction 
print(clf.predict(test_data))

#output 
[0 1 2]

# data visualization
from sklearn.externals.six import StringIO
import pydot
import graphviz
dot_data = StringIO()
tree.export_graphviz(
    clf,
    out_file = dot_data,               
    feature_names = iris.feature_names,
    class_names = iris.target_names,
    filled = True, rounded = True,
    impurity = False)
graph = pydot.graph_from_dot_data(dot_data.getvalue)
graph.write_pdf("iris.pdf")
print(graph)

print(test_data[0], test_target[0])
# output
[5.1 3.5 1.4 0.2] 0

print(iris.feature_names, iris.target_names)
#output for comparison
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] ['setosa' 'versicolor' 'virginica']

