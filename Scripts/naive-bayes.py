import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve # Adicionado learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt # Adicionado matplotlib

# 1. Carregar o Dataset
try:
    # Adapte o caminho para o seu arquivo CSV gerado
    df = pd.read_csv("/home/nicolas-amaral/Repositories/color-harmony-helper/Datasets/dataset_rgb_rotulado1.csv") #
except FileNotFoundError:
    print("Arquivo do dataset não encontrado. Verifique o caminho.")
    exit()

print("Dataset original carregado:")
print(df.head(6))

# 2. Preparação dos Dados para Grupos
grouped = df.groupby('GRUPO')
features_list = []
labels_list = []

for group_name, group_data in grouped:
    if len(group_data) == 3:
        feature_row = group_data[['R', 'G', 'B']].values.flatten().tolist()
        features_list.append(feature_row)
        label = 1 if group_data['RÓTULO'].iloc[0] == 'harmonico' else 0 #
        labels_list.append(label)

X_grouped = pd.DataFrame(features_list)
y_grouped = pd.Series(labels_list)

if X_grouped.shape[1] == 9:
    X_grouped.columns = ['R1', 'G1', 'B1', 'R2', 'G2', 'B2', 'R3', 'G3', 'B3']

print("\nDataset processado para grupos:")
print(X_grouped.head())
print("\nRótulos dos grupos:")
print(y_grouped.head())
print(f"\nDistribuição das classes dos grupos:\n{y_grouped.value_counts(normalize=True)}")

# 3. Treinamento do Modelo Gaussian Naive Bayes
model = GaussianNB() #

# ---------------------------------------------------------------------------
# 3.1 GERAR DADOS PARA CURVA DE APRENDIZADO E PLOTAR
# ---------------------------------------------------------------------------
# Definir os tamanhos das porções do dataset de treinamento a serem usadas
# np.linspace(0.1, 1.0, 10) significa usar 10 passos, de 10% a 100% do dataset
train_sizes_abs, train_scores, test_scores = learning_curve(
    model,
    X_grouped,
    y_grouped,
    cv=KFold(n_splits=5, shuffle=True, random_state=42), # Usar a mesma estratégia de CV
    scoring='accuracy',
    n_jobs=-1, # Usar todos os processadores disponíveis
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42 # Para reprodutibilidade da função learning_curve
)

# Calcular médias e desvios padrão para as pontuações de treinamento
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

# Calcular médias e desvios padrão para as pontuações de validação cruzada (teste)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Armazenar os resultados (você pode salvar em arquivos CSV/NPY se precisar)
learning_curve_data = {
    "train_sizes_abs": train_sizes_abs,
    "train_scores_mean": train_scores_mean,
    "train_scores_std": train_scores_std,
    "test_scores_mean": test_scores_mean,
    "test_scores_std": test_scores_std
}
print("\nDados da Curva de Aprendizado Coletados.")
# Exemplo: print(pd.DataFrame(learning_curve_data)) # Para ver os dados tabulados

# Plotar a curva de aprendizado
plt.figure(figsize=(10, 6))
plt.title("Curva de Aprendizado (Gaussian Naive Bayes)")
plt.xlabel("Número de Amostras de Treinamento")
plt.ylabel("Pontuação (Acurácia)")
plt.grid(True)

plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Pontuação de Treinamento")
plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Pontuação de Validação Cruzada")

plt.legend(loc="best")
plt.show() # Exibe o gráfico. Para salvar: plt.savefig('curva_aprendizado.png')

# ---------------------------------------------------------------------------
# 4. Validação do Modelo (usando cross_val_score para uma avaliação geral)
# ---------------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42) #
cv_results = cross_val_score(model, X_grouped, y_grouped, cv=kf, scoring='accuracy') #

print(f"\nAcurácia da Validação Cruzada (k=5) para grupos: {cv_results}")
print(f"Acurácia Média: {np.mean(cv_results):.4f}")
print(f"Desvio Padrão da Acurácia: {np.std(cv_results):.4f}")

if np.mean(cv_results) >= 0.95: #
    print("Acurácia média na validação cruzada ATINGIU o mínimo de 95%.") #
else:
    print("ALERTA: Acurácia média na validação cruzada ABAIXO do mínimo de 95%.") #

# Para calcular Taxa de Falsos Positivos e Negativos (usando uma divisão treino/teste)
X_train, X_test, y_train, y_test = train_test_split(X_grouped, y_grouped, test_size=0.2, random_state=42, stratify=y_grouped)

# É importante treinar um novo modelo aqui para a avaliação em X_test,
# ou usar o 'model' que já foi treinado pela learning_curve na última iteração (com todos os dados).
# Para consistência e avaliação da capacidade de generalização, vamos treinar um modelo com X_train.
model_for_report = GaussianNB()
model_for_report.fit(X_train, y_train)
y_pred = model_for_report.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred)
print(f"\nAcurácia no conjunto de teste (grupos): {accuracy_test:.4f}")

print("\nRelatório de Classificação (grupos):")
print(classification_report(y_test, y_pred, target_names=['nao_harmonico', 'harmonico'])) #

print("\nMatriz de Confusão (grupos):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

tn, fp, fn, tp = cm.ravel()
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"Taxa de Falsos Positivos (FPR) para grupos: {false_positive_rate:.4f}") #
print(f"Taxa de Falsos Negativos (FNR) para grupos: {false_negative_rate:.4f}") #

if false_positive_rate <= 0.05: #
    print("Taxa de Falsos Positivos DENTRO do limite de 5%.") #
else:
    print("ALERTA: Taxa de Falsos Positivos ACIMA do limite de 5%.") #

if false_negative_rate <= 0.05: #
    print("Taxa de Falsos Negativos DENTRO do limite de 5%.") #
else:
    print("ALERTA: Taxa de Falsos Negativos ACIMA do limite de 5%.") #

# 5. Treinar o modelo com todo o dataset de grupos para uso final
if np.mean(cv_results) >= 0.95 and false_positive_rate <= 0.05 and false_negative_rate <= 0.05: #
    print("\nTreinando o modelo final com todos os dados de GRUPOS...")
    final_model_grouped = GaussianNB() #
    final_model_grouped.fit(X_grouped, y_grouped)
    print("Modelo final (para grupos) treinado e pronto para ser salvo/usado.")
    # import joblib
    # joblib.dump(final_model_grouped, 'modelo_naive_bayes_grupos_cores.pkl')
else:
    print("\nO modelo (para grupos) não atendeu a todos os critérios de validação. Revisar dataset ou parâmetros.")