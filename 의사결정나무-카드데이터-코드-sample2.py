
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


import seaborn as sns

dataset.columns

plt.figure(figsize=(7, 7))
sns.scatterplot(x= '유동인구', y= '거주인구', hue='판매수준', style='판매수준', s=100, data=dataset)
plt.show()


dataset_new = pd.DataFrame()
dataset_new['읍면동명'] = dataset['읍면동명']
dataset_new['업종명'] = dataset['업종명']
dataset_new['성별'] = dataset['성별']


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import tree

#타이타닉 의사결정나무 => https://wikidocs.net/50114 => 범주형 변환 -> 연속형으로...

enc_classes = {} 
def encoding_label(x):   # x: 범주형 타입의 컬럼(Series)
    le = LabelEncoder()
    le.fit(x)
    label = le.transform(x)
    enc_classes[x.name] = le.classes_   # x.name: 컬럼명
    return label

dataset_new = dataset_new.apply(encoding_label)
dataset_new.head()

dataset_new['거주인구'] = dataset['거주인구']
dataset_new['유동인구'] = dataset['유동인구']
dataset_new['연령대'] = dataset['연령대']
dataset_new['판매수준(학21)'] = dataset['판매수준']
dataset_new = dataset_new.dropna()
dataset_new.dtypes


## getting X, y values
X = dataset_new[['읍면동명', '업종명', '성별', '거주인구', '유동인구', '연령대' ]]
y = dataset_new['판매수준(학21)']

## initiating DecisionTreeClassifer method
dt_clf = DecisionTreeClassifier(random_state = 1004)
'''
dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=6, max_leaf_nodes=None, 
                                   min_samples_split=2, min_samples_leaf=1, max_features=None)'''

## fitting a decision tree classifier
dt_clf_model = dt_clf.fit(X, y)


## feature importances
dt_clf_model.feature_importances_

#array([0.43809524, 0.56190476])

dt_clf_model_text = tree.export_text(dt_clf_model)

print(dt_clf_model_text)

fig = plt.figure(figsize=(15, 8))
_ = tree.plot_tree(dt_clf_model, 
                  feature_names=['읍면동명', '업종명', '성별', '거주인구', '유동인구', '연령대' ],
                  class_names=['상', '중', '하'],
                  filled=True)


## Visualizing Tree using Graphviz
from sklearn import tree
import graphviz

## exporting tree in DOT format
## refer to: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
tree_dot = tree.export_graphviz(
    dt_clf_model, 
    feature_names=['읍면동명', '업종명', '성별', '거주인구', '유동인구', '연령대' ],
    class_names=['상', '중', '하'],
    filled=True
)


## draw graph using Graphviz
dt_graph = graphviz.Source(tree_dot, format='png')
dt_graph
#에러 발생시 => CalledProcessError: Command '[WindowsPath('dot'), '-Kdot', '-Tsvg']' returned non-zero exit status 1.
# => conda install graphviz  

