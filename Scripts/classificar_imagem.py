import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tkinter import Tk, filedialog
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from itertools import combinations
from PIL import Image
import colorsys

# Carregar o modelo e os arquivos de pré-processamento salvos
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
modelo_dir = os.path.join(base_dir, "Modelo")

modelo_path = os.path.join(modelo_dir, "modelo_naive_bayes_treinado.pkl")
scaler_path = os.path.join(modelo_dir, "scaler_naive.pkl")
selector_path = os.path.join(modelo_dir, "selector_naive.pkl")

modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)
selector = joblib.load(selector_path)

# Pedir ao usuário para selecionar uma imagem
Tk().withdraw()
caminho_imagem = filedialog.askopenfilename(title="Selecione uma imagem",
                                             filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
if not caminho_imagem:
    print("Nenhuma imagem selecionada.")
    exit()

imagem = Image.open(caminho_imagem)
imagem = imagem.convert("RGB")
imagem = imagem.resize((200, 200))  # Reduz o tamanho da imagem para facilitar o processamento

# Extraímos as cores predominantes da imagem, descartando o fundo branco
pixels = np.array(imagem).reshape(-1, 3)

# Remover pixels muito próximos ao branco (qualquer cor muito clara será descartada)
limiar_branco = 240  # Ajuste para descartar pixels próximos ao branco
pixels_filtrados = pixels[np.all(pixels < limiar_branco, axis=1)]

# Usar KMeans para encontrar as 5 cores predominantes
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels_filtrados)
cores_rgb = kmeans.cluster_centers_.astype(int)

# Convertendo as cores para o espaço LAB para facilitar a análise de harmonia
cores_rgb_normalizadas = cores_rgb[np.newaxis, :, :] / 255.0
cores_lab = rgb2lab(cores_rgb_normalizadas)[0].flatten()

# Função para extrair características das cores
def extrair_features_lab(lab):
    df_lab = pd.DataFrame([lab], columns=[f'{c}{i}' for i in range(1,6) for c in ['L','a','b']])
    features = pd.DataFrame()
    
    # Calcula a média e o desvio padrão das componentes LAB
    features['L_mean'] = df_lab[[f'L{i}' for i in range(1,6)]].mean(axis=1)
    features['a_mean'] = df_lab[[f'a{i}' for i in range(1,6)]].mean(axis=1)
    features['b_mean'] = df_lab[[f'b{i}' for i in range(1,6)]].mean(axis=1)
    features['L_std'] = df_lab[[f'L{i}' for i in range(1,6)]].std(axis=1)
    features['a_std'] = df_lab[[f'a{i}' for i in range(1,6)]].std(axis=1)
    features['b_std'] = df_lab[[f'b{i}' for i in range(1,6)]].std(axis=1)
    
    # Calcula a distância entre as cores
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
    
    # Adiciona uma coluna de distâncias médias entre as cores
    features['media_dist'] = features[dist_cols].mean(axis=1)
    features['raz_dist_1_5_1_2'] = features['dist_1_5'] / (features['dist_1_2'] + 1e-5)
    
    return pd.concat([df_lab, features], axis=1)

X_full = extrair_features_lab(cores_lab)

# Normalizando os dados para o modelo
X_scaled = scaler.transform(X_full)
X_selected = selector.transform(X_scaled)

# Calculando a probabilidade de harmonia com base no modelo
threshold = 0.30  # Threshold mínimo para considerar uma imagem harmônica
proba = modelo.predict_proba(X_selected)[0, 1]
harmonico = proba >= threshold

# Função para gerar uma paleta harmônica de cores
def gerar_paleta_harmonica(cor_principal_rgb):
    cor_principal_hls = colorsys.rgb_to_hls(cor_principal_rgb[0]/255.0, cor_principal_rgb[1]/255.0, cor_principal_rgb[2]/255.0)
    
    cores_sugeridas = []

    # Gerar cores complementares, análogas e triádicas
    h_complementar = (cor_principal_hls[0] + 0.5) % 1.0
    cores_sugeridas.append(colorsys.hls_to_rgb(h_complementar, cor_principal_hls[1], cor_principal_hls[2]))

    for i in [30, 60]:
        h_analogas1 = (cor_principal_hls[0] + i / 360.0) % 1.0
        h_analogas2 = (cor_principal_hls[0] - i / 360.0) % 1.0
        cores_sugeridas.append(colorsys.hls_to_rgb(h_analogas1, cor_principal_hls[1], cor_principal_hls[2]))
        cores_sugeridas.append(colorsys.hls_to_rgb(h_analogas2, cor_principal_hls[1], cor_principal_hls[2]))

    for i in [120, 240]:
        h_triadica = (cor_principal_hls[0] + i / 360.0) % 1.0
        cores_sugeridas.append(colorsys.hls_to_rgb(h_triadica, cor_principal_hls[1], cor_principal_hls[2]))

    cores_sugeridas = [tuple(int(c * 255) for c in color) for color in cores_sugeridas]
    
    return cores_sugeridas[:5]  # Devolve as 5 cores mais harmônicas

# Ajustando as cores da paleta até atingir 80% de harmonia
def ajustar_paleta(cores_rgb, proba):
    paleta_harmonica = cores_rgb.copy()
    i = 0
    while proba < 0.80 and i < len(paleta_harmonica):
        # Ajusta uma cor por vez, procurando por alternativas harmônicas
        paleta_harmonica[i] = gerar_paleta_harmonica(paleta_harmonica[i])[0]
        X_full = extrair_features_lab(rgb2lab(np.array(paleta_harmonica)[np.newaxis, :, :]/255.0).flatten())
        X_scaled = scaler.transform(X_full)
        X_selected = selector.transform(X_scaled)
        proba = modelo.predict_proba(X_selected)[0, 1]  # Atualiza a probabilidade após o ajuste
        i += 1
    return paleta_harmonica, proba

# Lógica para ajustar a paleta dependendo da probabilidade de harmonia
if proba >= 0.80:
    print("A imagem é muito harmônica!")
    paleta_harmonica = cores_rgb[:5]  # Usa as cores dominantes sem alterações
elif proba >= 0.51:
    print("A imagem tem uma harmonia moderada, ajustando as cores para torná-la mais harmônica...")
    paleta_harmonica, proba = ajustar_paleta(cores_rgb, proba)
else:
    print("A imagem não é harmônica, sugerindo uma paleta harmônica de cores...")
    paleta_harmonica = gerar_paleta_harmonica(cores_rgb[0])

print(f"Probabilidade de harmonia: {proba:.2f}")

# Exibir as cores dominantes e a paleta sugerida em gráficos
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Exibir as cores dominantes da imagem
ax[0].imshow([cores_rgb / 255.0])  # Normalizando para o intervalo [0, 1]
ax[0].axis('off')
ax[0].set_title("Cores Dominantes da Imagem")

# Normalizar as cores da paleta harmônica para [0, 1] antes de exibir
paleta_harmonica_normalizada = [np.array(cor) / 255.0 for cor in paleta_harmonica]

# Exibir a paleta harmônica sugerida
ax[1].imshow([paleta_harmonica_normalizada])
ax[1].axis('off')
ax[1].set_title("Paleta Harmônica Sugerida")

plt.show()
