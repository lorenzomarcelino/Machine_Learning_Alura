import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20
uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/"
"\n16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)

mapa = {
    'expected_hours' : 'horas_esperadas',
    'price' : 'preco',
    'unfinished' : 'nao_finalizado'
}
dados = dados.rename(columns = mapa)
troca = {
    0 : 1,
    1 : 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

sns.scatterplot(x="horas_esperadas", y="preco", hue="finalizado", data=dados)
sns.relplot(x="horas_esperadas", y="preco", hue="finalizado", col="finalizado", data = dados)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size =0.25, stratify=y)

total = len(treino_x) + len(teste_x)
proporcao_treino_x = len(treino_x) / total
proporcao_teste_x = len(teste_x) / total

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))
print("treinamos com %0.2f%% dos dados e testamos com %0.2f%%" % (proporcao_treino_x * 100, proporcao_teste_x * 100))


modelo = LinearSVC (random_state = SEED)
modelo. fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

print("Taxa de acerto: %.2f%%" % (accuracy_score(teste_y, previsoes) * 100))

baseline = np.ones(540)
acuracia = accuracy_score(teste_y, baseline) * 100
print("Taxa de acerto (Baseline): %.2f%%" % acuracia)
