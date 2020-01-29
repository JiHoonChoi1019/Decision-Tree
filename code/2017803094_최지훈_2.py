#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

#Dot to png
import pydot

iris = load_iris()
X = iris.data[:, 2:] #Petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X,y)

from sklearn.tree import export_graphviz

export_graphviz(tree_clf, out_file='iris_tree.dot', 
                feature_names=['petal length(cm)', 'petal width(cm)'],
               class_names = iris.target_names,
               rounded=True,
               filled=True)


#Encoding 중요
(graph,) = pydot.graph_from_dot_file('iris_tree.dot', encoding='utf8')

#Dot 파일을 Png 이미지로 저장
graph.write_png('iris_tree.png')


# In[ ]:





# In[ ]:





# In[ ]:




