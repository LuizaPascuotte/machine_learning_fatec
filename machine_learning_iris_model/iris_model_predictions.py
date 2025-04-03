import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris # Importação do conjunto de dados Iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 1. Carregar o conjunto de dados Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


# Definir diferentes conjuntos de features
features_all = data.feature_names  # Todos os atributos
features_width = ["sepal width (cm)", "petal width (cm)"]  # Apenas larguras
features_length = ["sepal length (cm)", "petal length (cm)"]  # Apenas comprimentos

# Função para treinar e avaliar o modelo
def train_and_evaluate(features, name):
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{name} - Tabela de Confusão:")
    print(cm)
    print(f"\n{name} - Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Visualizar a Matriz de Confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão - {name}')
    plt.show()

# Treinar e avaliar os modelos
train_and_evaluate(features_all, "Modelo com todos os atributos")
train_and_evaluate(features_width, "Modelo com atributos de Largura")
train_and_evaluate(features_length, "Modelo com atributos de Comprimento")
