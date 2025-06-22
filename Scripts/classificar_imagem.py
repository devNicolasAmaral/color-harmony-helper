import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tkinter import Tk, filedialog
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
from itertools import combinations
from PIL import Image

# 📁 Etapa 1: Localizar arquivos salvos
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
modelo_dir = os.path.join(base_dir, "Modelo")

modelo_path = os.path.join(modelo_dir, "modelo_naive_bayes_treinado.pkl")
scaler_path = os.path.join(modelo_dir, "scaler_naive.pkl")
selector_path = os.path.join(modelo_dir, "selector_naive.pkl")

modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)
selector = joblib.load(selector_path)

# 📤 Etapa 2: Upload da imagem
Tk().withdraw()
caminho_imagem = filedialog.askopenfilename(title="Selecione uma imagem",
                                             filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
if not caminho_imagem:
    print("❌ Nenhuma imagem selecionada.")
    exit()

imagem = Image.open(caminho_imagem)
imagem = imagem.convert("RGB")
imagem = imagem.resize((200, 200))  # Reduz o tamanho para facilitar

# 🎨 Etapa 3: Extrair as 5 cores dominantes
pixels = np.array(imagem).reshape(-1, 3)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels)
cores_rgb = kmeans.cluster_centers_.astype(int)

# 🔁 Etapa 4: Converter para LAB
cores_rgb_normalizadas = cores_rgb[np.newaxis, :, :] / 255.0
cores_lab = rgb2lab(cores_rgb_normalizadas)[0].flatten()

# 🧪 Etapa 5: Extrair features da paleta
def extrair_features_lab(lab):
    df_lab = pd.DataFrame([lab], columns=[f'{c}{i}' for i in range(1,6) for c in ['L','a','b']])
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
    return pd.concat([df_lab, features], axis=1)

X_full = extrair_features_lab(cores_lab)

# ⚙️ Etapa 6: Padronizar e selecionar features
X_scaled = scaler.transform(X_full)
X_selected = selector.transform(X_scaled)

# 🧠 Etapa 7: Classificação com threshold
threshold = 0.35
proba = modelo.predict_proba(X_selected)[0, 1]
harmonico = proba >= threshold

# 🎯 Etapa 8: Substituir se não harmonico
def gerar_alternativa(cor):
    cor = np.array(cor)
    nova = np.clip(cor + np.random.randint(-15, 15, size=3), 0, 255)
    return nova

nova_paleta = cores_rgb.copy()
if not harmonico:
    for i in range(len(nova_paleta)):
        nova_paleta[i] = gerar_alternativa(nova_paleta[i])
    print("⚠️ Paleta ajustada para ser mais harmônica.")
else:
    print("✅ A paleta extraída já é harmônica!")

# 🖼️ Etapa 9: Mostrar paletas lado a lado
def mostrar_paletas_juntas(original, ajustada):
    fig, axs = plt.subplots(2, len(original), figsize=(12, 3))
    
    for i in range(len(original)):
        axs[0, i].imshow([[original[i].astype(int)]])
        axs[0, i].axis('off')
        axs[1, i].imshow([[ajustada[i].astype(int)]])
        axs[1, i].axis('off')

    axs[0, 0].set_ylabel("Original", fontsize=12)
    axs[1, 0].set_ylabel("Ajustada", fontsize=12)

    plt.suptitle("Comparação de Paletas", fontsize=14)
    plt.tight_layout()
    plt.show()

mostrar_paletas_juntas(cores_rgb, nova_paleta)