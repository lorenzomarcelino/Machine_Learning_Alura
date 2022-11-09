from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

dados = pd.read_csv('/content/Customer-Churn.csv') 

traducao_dic = {'Sim': 1, 
                'Nao': 0}

dadosmodificados = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao_dic)

dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'], axis=1))

dados_final = pd.concat([dadosmodificados, dummie_dados], axis=1)

Xmaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]

X = dados_final.drop('Churn', axis = 1)
y = dados_final['Churn']
smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y)
dados_final = pd.concat([X, y], axis=1)

norm = StandardScaler()
X_normalizado = norm.fit_transform(X)
Xmaria_normalizado = norm.transform(pd.DataFrame(Xmaria, columns = X.columns))

X_treino, X_teste, y_treino, y_teste = train_test_split(X_normalizado, y, test_size=0.3, random_state=123)

knn = KNeighborsClassifier(metric='euclidean')
knn.fit(X_treino, y_treino)
predito_knn = knn.predict(X_teste)

bnb = BernoulliNB(binarize=-0.44)
np.median(X_treino)
bnb.fit(X_treino, y_treino)
predito_BNb = bnb.predict(X_teste)

dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)
dtc.fit(X_treino, y_treino)
predito_ArvoreDecisao = dtc.predict(X_teste)

print('Modelo KNN: ', precision_score(y_teste, predito_knn))
print('Modelo Bernoulli de Naive Bayes: ', precision_score(y_teste, predito_BNb))
print('Modelo Árvore de Decisão: ', precision_score(y_teste, predito_ArvoreDecisao))
print('Recall Modelo KNN: ', recall_score(y_teste, predito_knn))
print('Recall Modelo Bernoulli de Naive Bayes: ', recall_score(y_teste, predito_BNb))
print('Recall Modelo Árvore de Decisão: ', recall_score(y_teste, predito_ArvoreDecisao))
