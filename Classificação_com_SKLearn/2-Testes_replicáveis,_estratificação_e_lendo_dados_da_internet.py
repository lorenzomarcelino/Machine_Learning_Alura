import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20
uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/"
"\n10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)

mapa = { 
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"
}
dados = dados.rename(columns = mapa)

x= dados[["principal","como_funciona","contato"]]
y= dados["comprou"]

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)

print("treinamos com %0.2f%% dos dados e testamos com %0.2f%%" % (len(treino_x) * 100 / 99, len(teste_x) * 100 / 99))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

print("Taxa de acerto: %.2f%%" % (accuracy_score(teste_y, previsoes) * 100))
