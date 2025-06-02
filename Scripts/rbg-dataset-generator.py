import pandas as pd
import random
import os

# Diretório onde os arquivos serão salvos
diretorio_saida = "/home/nicolas-amaral/Repositories/color-harmony-helper/Datasets"

# Função para gerar cores harmônicas
def gerar_grupo_harmonico():
    base = [random.randint(0, 255) for _ in range(3)]
    return [
        base,
        [min(255, base[0] + random.randint(-10, 10)), min(255, base[1] + random.randint(-10, 10)), min(255, base[2] + random.randint(-10, 10))],
        [min(255, base[0] + random.randint(-20, 20)), min(255, base[1] + random.randint(-20, 20)), min(255, base[2] + random.randint(-20, 20))],
    ]

# Função para gerar cores não harmônicas
def gerar_grupo_nao_harmonico():
    return [
        [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
        [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
        [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
    ]

# Função para gerar X linhas de cores com variação na proporção
def gerar_cores(linhas=100000, proporcao_min_harmonicos=0.4, proporcao_min_nao_harmonicos=0.4):
    total_grupos = linhas // 3

    # Adiciona variação nas proporções
    proporcao_harmonicos = random.uniform(proporcao_min_harmonicos, 1 - proporcao_min_nao_harmonicos)
    proporcao_nao_harmonicos = 1 - proporcao_harmonicos

    grupos_harmonicos = int(total_grupos * proporcao_harmonicos)
    grupos_nao_harmonicos = int(total_grupos * proporcao_nao_harmonicos)

    cores = []
    rotulos = []

    # Gera grupos harmônicos
    for _ in range(grupos_harmonicos):
        cores.extend(gerar_grupo_harmonico())
        rotulos.extend(["harmonico"] * 3)

    # Gera grupos não harmônicos
    for _ in range(grupos_nao_harmonicos):
        cores.extend(gerar_grupo_nao_harmonico())
        rotulos.extend(["nao_harmonico"] * 3)

    return cores, rotulos, proporcao_harmonicos, proporcao_nao_harmonicos

# Função para calcular porcentagens de harmonia
def calcular_porcentagem_harmonia(rotulos):
    total_grupos = len(rotulos) // 3
    harmonicos = rotulos.count("harmonico") // 3
    nao_harmonicos = rotulos.count("nao_harmonico") // 3

    perc_harmonicos = (harmonicos / total_grupos) * 100
    perc_nao_harmonicos = (nao_harmonicos / total_grupos) * 100

    print(f"Total de grupos: {total_grupos}")
    print(f"Grupos harmônicos: {perc_harmonicos:.2f}%")
    print(f"Grupos não harmônicos: {perc_nao_harmonicos:.2f}%")

# Função para determinar o próximo nome do arquivo
def determinar_nome_arquivo(diretorio, base_nome="dataset_rgb_rotulado"):
    arquivos_existentes = os.listdir(diretorio)
    numeros = []

    for arquivo in arquivos_existentes:
        if arquivo.startswith(base_nome) and arquivo.endswith(".csv"):
            try:
                numero = int(arquivo.replace(base_nome, "").replace(".csv", ""))
                numeros.append(numero)
            except ValueError:
                pass

    proximo_numero = max(numeros) + 1 if numeros else 1
    return f"{base_nome}{proximo_numero}.csv"

# Gera cores e rótulos com variação nas proporções
cores, rotulos, proporcao_harmonicos, proporcao_nao_harmonicos = gerar_cores(linhas=10000, proporcao_min_harmonicos=0.4, proporcao_min_nao_harmonicos=0.4)

# Cria DataFrame
df = pd.DataFrame(cores, columns=["R", "G", "B"])
df['RÓTULO'] = rotulos

# Adiciona coluna de grupos
df['GRUPO'] = [i // 3 + 1 for i in range(len(df))]

# Calcula e exibir as porcentagens de harmonia
calcular_porcentagem_harmonia(rotulos)

# Determina o próximo nome do arquivo
nome_arquivo = determinar_nome_arquivo(diretorio_saida)

# Salva o novo 
caminho_saida = os.path.join(diretorio_saida,  f"{nome_arquivo}")
df.to_csv(caminho_saida, index=False)

print(f"Arquivo salvo em: {caminho_saida}")