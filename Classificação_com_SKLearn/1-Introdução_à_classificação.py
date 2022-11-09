from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features 
# pelo longo?
# perna curta?
# faz Au-Au?

# porco -> 1
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]
# cachorro -> 0
cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0]

model = LinearSVC()
model.fit(treino_x, treino_y)

misterioso1 = [1, 1, 1]
misterioso2 = [1, 1, 0]
misterioso3 = [0, 1, 1]

teste_x = [misterioso1, misterioso2, misterioso3]
teste_y = [0, 1, 1]
previsoes = model.predict(teste_x)

taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto: %.2f" % taxa_de_acerto)
