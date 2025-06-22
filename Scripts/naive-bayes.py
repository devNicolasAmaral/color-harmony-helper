import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from itertools import combinations
import seaborn as sns
import joblib

print("📁 Etapa 1: Localizando o dataset mais recente...")
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
datasets_dir = os.path.join(base_dir, "Datasets")
arquivos_csv = [f for f in os.listdir(datasets_dir) if re.match(r'dataset_rgb_rotulado\d+\.csv', f)]
if not arquivos_csv:
    raise FileNotFoundError("❌ Nenhum arquivo encontrado.")
ultimo_csv = sorted(arquivos_csv, key=lambda x: int(re.findall(r'\d+', x)[0]))[-1]
caminho_csv = os.path.join(datasets_dir, ultimo_csv)
print(f"✅ Usando o dataset: {caminho_csv}")

print("📥 Etapa 2: Lendo e agrupando os dados...")
df = pd.read_csv(caminho_csv).dropna().copy()
df["GRUPO"] = df["GRUPO"].astype(int)

def agrupar_cores(df):
    grupos = df['GRUPO'].unique()
    dados, rotulos = [], []
    for g in grupos:
        grupo_df = df[df['GRUPO'] == g].reset_index(drop=True)
        if len(grupo_df) < 5:
            continue
        grupo_df = grupo_df.iloc[:5]
        linha = []
        for i in range(5):
            linha.extend([grupo_df.loc[i, 'R'], grupo_df.loc[i, 'G'], grupo_df.loc[i, 'B']])
        dados.append(linha)
        rotulos.append(1 if grupo_df.loc[0, 'RÓTULO'] == 'harmonico' else 0)
    return pd.DataFrame(dados, columns=[f'{c}{i}' for i in range(1,6) for c in ['R','G','B']]), rotulos

X_rgb, y_bin = agrupar_cores(df)
print(f"✅ Grupos válidos encontrados: {len(X_rgb)}")
if len(X_rgb) == 0:
    raise ValueError("❌ Nenhum grupo válido encontrado.")

print("🎨 Etapa 3: Convertendo RGB para LAB...")
def rgb_quintuples_to_lab(df_rgb):
    labs = []
    for _, row in df_rgb.iterrows():
        rgb_colors = np.array([
            [row[f'R{i}'], row[f'G{i}'], row[f'B{i}']] for i in range(1,6)
        ], dtype=np.uint8)[np.newaxis, :, :] / 255.0
        lab_colors = rgb2lab(rgb_colors)[0]
        labs.append(lab_colors.flatten())
    cols_lab = [f'{c}{i}' for i in range(1,6) for c in ['L','a','b']]
    return pd.DataFrame(labs, columns=cols_lab)

X_lab = rgb_quintuples_to_lab(X_rgb)

print("🧪 Etapa 4: Extraindo features...")
def extrair_features(df_lab):
    features = pd.DataFrame()
    features['L_mean'] = df_lab[[f'L{i}' for i in range(1,6)]].mean(axis=1)
    features['a_mean'] = df_lab[[f'a{i}' for i in range(1,6)]].mean(axis=1)
    features['b_mean'] = df_lab[[f'b{i}' for i in range(1,6)]].mean(axis=1)
    features['L_std'] = df_lab[[f'L{i}' for i in range(1,6)]].std(axis=1)
    features['a_std'] = df_lab[[f'a{i}' for i in range(1,6)]].std(axis=1)
    features['b_std'] = df_lab[[f'b{i}' for i in range(1,6)]].std(axis=1)

    def dist(l1, a1, b1, l2, a2, b2):
        return np.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)

    dist_cols = []
    for i, j in combinations(range(1,6), 2):
        col = f'dist_{i}_{j}'
        features[col] = dist(
            df_lab[f'L{i}'], df_lab[f'a{i}'], df_lab[f'b{i}'],
            df_lab[f'L{j}'], df_lab[f'a{j}'], df_lab[f'b{j}']
        )
        dist_cols.append(col)

    features['media_dist'] = features[dist_cols].mean(axis=1)
    features['raz_dist_1_5_1_2'] = features['dist_1_5'] / (features['dist_1_2'] + 1e-5)
    return features

X_features = extrair_features(X_lab)
X_final = pd.concat([X_lab, X_features], axis=1)

print("⚙️ Etapa 5: Padronizando e selecionando as melhores features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y_bin)
feature_names = X_final.columns[selector.get_support()]

print("📊 Etapa 6: Curva de aprendizado...")
model = GaussianNB()
train_sizes, train_scores, test_scores = learning_curve(
    model, X_selected, y_bin,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

plt.figure(figsize=(10,6))
plt.title("Curva de Aprendizado (GaussianNB)")
plt.xlabel("Tamanho do Treinamento")
plt.ylabel("Acurácia")
plt.grid(True)
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Treino")
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Validação")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

print("📈 Etapa 7: Validação cruzada...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(GaussianNB(), X_selected, y_bin, cv=cv, scoring='accuracy')
print(f"Acurácias (CV): {cv_scores}")
print(f"Acurácia média: {np.mean(cv_scores):.4f}")
print(f"Desvio padrão: {np.std(cv_scores):.4f}")

print("🧠 Etapa 8: Avaliação final do modelo (com threshold = 0.35 e priors = [0.25, 0.75])...")
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

model = GaussianNB(priors=[0.25, 0.75])
model.fit(X_train, y_train)

threshold = 0.35
probas = model.predict_proba(X_test)
y_pred_adjusted = (probas[:, 1] >= threshold).astype(int)

print("\nRelatório de Classificação (com threshold e priors ajustados):")
print(classification_report(y_test, y_pred_adjusted, target_names=['nao_harmonico', 'harmonico']))

cm = confusion_matrix(y_test, y_pred_adjusted)
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

# Matriz de confusão visual
print("🧮 Matriz de Confusão Visual")
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['nao_harmonico', 'harmonico'])
disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
plt.title("Matriz de Confusão (threshold = 0.35, priors = [0.25, 0.75])")
plt.grid(False)
plt.tight_layout()
plt.show()

# 💾 Etapa Final: Salvando o modelo treinado
print("\n💾 Etapa Final: Salvando o modelo treinado...")

# Caminho relativo da pasta Scripts → para a pasta vizinha Modelo
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
modelo_dir = os.path.join(base_dir, "Modelo")
os.makedirs(modelo_dir, exist_ok=True)

caminho_modelo = os.path.join(modelo_dir, "modelo_naive_bayes_treinado.pkl")

# Salva apenas o modelo final treinado com priors
joblib.dump(model, caminho_modelo)

print(f"✅ Modelo salvo com sucesso em: {caminho_modelo}")

# 💾 Salvando scaler e seletor de features
print("\n💾 Salvando scaler e seletor de features...")

# Criar pasta Modelo se não existir
os.makedirs(modelo_dir, exist_ok=True)

# Salvar scaler
caminho_scaler = os.path.join(modelo_dir, "scaler_naive.pkl")
joblib.dump(scaler, caminho_scaler)

# Salvar seletor de features
caminho_selector = os.path.join(modelo_dir, "selector_naive.pkl")
joblib.dump(selector, caminho_selector)

print(f"✅ Scaler salvo em: {caminho_scaler}")
print(f"✅ Seletor salvo em: {caminho_selector}")
