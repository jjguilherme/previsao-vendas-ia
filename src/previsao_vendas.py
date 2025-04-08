# src/previsao_vendas.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Criando dataset fictício (substitua por pd.read_csv('data/vendas.csv') se tiver dados reais)
data = {
    'mes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'promocao': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    'vendas': [200, 300, 210, 350, 220, 230, 320, 340, 240, 360, 250, 370]
}
df = pd.DataFrame(data)

# Preparando os dados
X = df[['mes', 'promocao']]  # Features
y = df['vendas']             # Target

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Avaliando o modelo
score = modelo.score(X_test, y_test)
print(f"Acurácia do modelo (R²): {score:.2f}")

# Fazendo previsões para o mês 13
novos_dados = pd.DataFrame({'mes': [13, 13], 'promocao': [0, 1]})
previsoes = modelo.predict(novos_dados)
print("Previsões de vendas para o mês 13:")
for i, pred in enumerate(previsoes):
    promocao = "sem promoção" if novos_dados['promocao'][i] == 0 else "com promoção"
    print(f"Mês 13 {promocao}: {pred:.2f} vendas")

# Salvando previsões em CSV
resultado = novos_dados.copy()
resultado['vendas_previstas'] = previsoes
resultado.to_csv('previsoes_vendas.csv', index=False)
print("Previsões salvas em 'previsoes_vendas.csv'")