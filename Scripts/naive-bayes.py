import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from skimage.color import rgb2lab

# 1. Carregar o Dataset e reorganizar por grupo
df = pd.read_csv("C:/Datasets/dataset_rgb_rotulado1.csv")

def agrupar_cores(df):
    grupos = df['GRUPO'].unique()
    dados, rotulos = [], []
    for g in grupos:
        grupo_df = df[df['GRUPO'] == g].reset_index(drop=True)
        if len(grupo_df) != 3: continue
        linha = []
        for i in range(3):
            linha.extend([grupo_df.loc[i, 'R'], grupo_df.loc[i, 'G'], grupo_df.loc[i, 'B']])
        dados.append(linha)
        rotulos.append(1 if grupo_df.loc[0, 'RÓTULO'] == 'harmonico' else 0)
    return pd.DataFrame(dados, columns=[f'{c}{i}' for i in range(1,4) for c in ['R','G','B']]), rotulos

X_rgb, y_bin = agrupar_cores(df)

# 2. Conversão de RGB para LAB 
def rgb_triplets_to_lab(df_rgb):
    labs = []
    for _, row in df_rgb.iterrows():
        rgb_colors = np.array([
            [row['R1'], row['G1'], row['B1']],
            [row['R2'], row['G2'], row['B2']],
            [row['R3'], row['G3'], row['B3']]
        ], dtype=np.uint8)[np.newaxis, :, :] / 255.0
        lab_colors = rgb2lab(rgb_colors)[0]
        labs.append(lab_colors.flatten())
    cols_lab = [f'{c}{i}' for i in range(1,4) for c in ['L','a','b']]
    return pd.DataFrame(labs, columns=cols_lab)

X_lab = rgb_triplets_to_lab(X_rgb)

# 3. Features Adicionais
def extrair_features(df_lab):
    features = pd.DataFrame()
    features['L_mean'] = df_lab[[f'L{i}' for i in range(1,4)]].mean(axis=1)
    features['a_mean'] = df_lab[[f'a{i}' for i in range(1,4)]].mean(axis=1)
    features['b_mean'] = df_lab[[f'b{i}' for i in range(1,4)]].mean(axis=1)

    def dist_lab(l1,a1,b1,l2,a2,b2):
        return np.sqrt((l1-l2)**2 + (a1-a2)**2 + (b1-b2)**2)

    features['dist_1_2'] = dist_lab(df_lab['L1'], df_lab['a1'], df_lab['b1'], df_lab['L2'], df_lab['a2'], df_lab['b2'])
    features['dist_2_3'] = dist_lab(df_lab['L2'], df_lab['a2'], df_lab['b2'], df_lab['L3'], df_lab['a3'], df_lab['b3'])
    features['dist_1_3'] = dist_lab(df_lab['L1'], df_lab['a1'], df_lab['b1'], df_lab['L3'], df_lab['a3'], df_lab['b3'])

    return features

X_features = extrair_features(X_lab)
X_final = pd.concat([X_lab, X_features], axis=1)

# 4. Gerar dados para Curva de Aprendizado
model = GaussianNB()
train_sizes, train_scores, test_scores = learning_curve(
    model, X_final, y_bin, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

plt.figure(figsize=(10,6))
plt.title("Curva de Aprendizado (GaussianNB)")
plt.xlabel("Tamanho do Treinamento")
plt.ylabel("Acurácia")
plt.grid(True)

plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                 np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color="r")
plt.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                 np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.1, color="g")

plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Treino")
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Validação")
plt.legend(loc="best")
plt.show()

# 5. Validação cruzada do modelo
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_final, y_bin, cv=cv, scoring='accuracy')

print(f"\nAcurácias (CV): {cv_scores}")
print(f"Acurácia média: {np.mean(cv_scores):.4f}")
print(f"Desvio padrão: {np.std(cv_scores):.4f}")

# 6. Avaliação Geral com conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X_final, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
model_final = GaussianNB()
model_final.fit(X_train, y_train)
y_pred = model_final.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['nao_harmonico', 'harmonico']))

cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)

tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"Taxa de Falsos Positivos (FPR): {fpr:.4f}")
print(f"Taxa de Falsos Negativos (FNR): {fnr:.4f}")

if np.mean(cv_scores) >= 0.95 and fpr <= 0.05 and fnr <= 0.05:
    print("✅ Modelo validado com sucesso e pronto para uso.")
else:
    print("⚠️ Modelo ainda não atendeu aos critérios desejados.")