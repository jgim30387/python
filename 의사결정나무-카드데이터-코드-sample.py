
#---------------------------------------
#2023-1학기 데이터마이닝 실습용 프로그램
#(탐색적 데이터분석)
#--------------------------------------

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn import model_selection
from sklearn import metrics
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

font_path = "c:/Windows/fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

dirname = 'C:/Users/aa/Downloads/의사결정나무모형구축-과제(데이터및sample코드포함)/'
dataset = pd.read_csv(dirname+ 'Card_data_sample.csv', encoding='cp949')
dataset.shape
dataset.info()


#이용금액을 상,중,하로 구분 
dataset.info()
dataset.describe()
temp = []
for i in range(len(dataset)):
    if dataset.iloc[i]['이용금액']>1.850000e+06:
        temp.append('상')
    elif dataset.iloc[i]['이용금액']>9.400000e+04:
        temp.append('중')
    else:
        temp.append('하')

       
dataset['판매수준'] = temp   
#dataset.to_csv(dirname + 'Card_data_new4.csv', encoding="cp949")

import seaborn as sns

dataset.columns

plt.figure(figsize=(7, 7))
sns.scatterplot(x= '유동인구', y= '거주인구', hue='판매수준', style='판매수준', s=100, data=dataset)
plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree

#타이타닉 의사결정나무 => https://wikidocs.net/50114 => 범주형 변환 -> 연속형으로...

dataset['판매수준']=dataset['판매수준'].replace(['하','중','상'],[0,1,2])
dataset['읍면동명']=dataset['읍면동명'].astype('category') 
dataset['업종명']=dataset['업종명'].astype('category')
dataset['관광구분']=dataset['관광구분'].astype('category') 
dataset['연령대']=dataset['연령대'].astype('category')
dataset['성별']=dataset['성별'].astype('category')
dataset1 = pd.get_dummies(dataset, columns = ['읍면동명', '업종명', '관광구분', '연령대','성별'])


## getting X, y values
x_data=dataset1.iloc[:,15:]
y_data=dataset1.iloc[:,14]
x_data = x_data.values
y_data = y_data.values

## initiating DecisionTreeClassifer method
#dt_clf = DecisionTreeClassifier(random_state = 1004)

dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=7, max_leaf_nodes=None, 
                                   min_samples_split=2, min_samples_leaf=1, max_features=None)

## fitting a decision tree classifier
dt_clf_model = dt_clf.fit(x_data, x_data)

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


from sklearn.tree import export_graphviz
export_graphviz(dt_clf_model, out_file="./tree.dot")


import graphviz
with open("./tree.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='tree', directory='./', cleanup=True)
#
#dot
import webbrowser
webbrowser.open('tree.png')


'''
## feature importances
dt_clf_model.feature_importances_

#array([0.43809524, 0.56190476])

dt_clf_model_text = tree.export_text(dt_clf_model)

print(dt_clf_model_text)

fig = plt.figure(figsize=(15, 8))
_ = tree.plot_tree(dt_clf_model, 
                  feature_names=['읍면동명', '업종명', '관광구분', '연령대','성별'],
                  class_names=[0,1,2],
                  filled=True)

## Visualizing Tree using Graphviz
from sklearn import tree
import graphviz

## exporting tree in DOT format
## refer to: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
tree_dot = tree.export_graphviz(
    dt_clf_model, 
    feature_names=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'], 
    class_names=['setosa', 'versicolor', 'virginica'],
    filled=True
)


## draw graph using Graphviz
dt_graph = graphviz.Source(tree_dot, format='png')
dt_graph
#에러 발생시 => CalledProcessError: Command '[WindowsPath('dot'), '-Kdot', '-Tsvg']' returned non-zero exit status 1.
# => conda install graphviz  

'''




