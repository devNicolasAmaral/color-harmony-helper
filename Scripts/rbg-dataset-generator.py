import pandas as pd
import random
import os

# Caminho para salvar na pasta Datasets (vizinha da pasta Scripts)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
diretorio_saida = os.path.join(base_dir, "Datasets")

# Função para gerar grupo harmônico
def gerar_grupo_harmonico():
    base = [random.randint(60, 195) for _ in range(3)]  # valores medianos para melhor contraste
    grupo = [base]
    for _ in range(4):  # total de 5 cores
        variacao = [max(0, min(255, base[i] + random.randint(-8, 8))) for i in range(3)]
        grupo.append(variacao)
    return grupo

# Função para gerar grupo não harmônico (totalmente aleatório)
def gerar_grupo_nao_harmonico():
    return [[random.randint(0, 255) for _ in range(3)] for _ in range(5)]

# Gerador principal
def gerar_cores(linhas=500000, proporcao_min_harmonicos=0.45, proporcao_min_nao_harmonicos=0.45):
    total_grupos = linhas // 5
    proporcao_harmonicos = random.uniform(proporcao_min_harmonicos, 1 - proporcao_min_nao_harmonicos)
    proporcao_nao_harmonicos = 1 - proporcao_harmonicos

    grupos_harmonicos = int(total_grupos * proporcao_harmonicos)
    grupos_nao_harmonicos = int(total_grupos * proporcao_nao_harmonicos)

    cores = []
    rotulos = []

    for _ in range(grupos_harmonicos):
        cores.extend(gerar_grupo_harmonico())
        rotulos.extend(["harmonico"] * 5)

    for _ in range(grupos_nao_harmonicos):
        cores.extend(gerar_grupo_nao_harmonico())
        rotulos.extend(["nao_harmonico"] * 5)

    return cores, rotulos, proporcao_harmonicos, proporcao_nao_harmonicos

# Estatísticas
def calcular_porcentagem_harmonia(rotulos):
    total_grupos = len(rotulos) // 5
    harmonicos = rotulos.count("harmonico") // 5
    nao_harmonicos = rotulos.count("nao_harmonico") // 5

    print(f"Total de grupos: {total_grupos}")
    print(f"Harmônicos: {harmonicos} ({(harmonicos / total_grupos) * 100:.2f}%)")
    print(f"Não harmônicos: {nao_harmonicos} ({(nao_harmonicos / total_grupos) * 100:.2f}%)")

# Nome automático para o arquivo
def determinar_nome_arquivo(diretorio, base_nome="dataset_rgb_rotulado"):
    arquivos = os.listdir(diretorio)
    numeros = []
    for arquivo in arquivos:
        if arquivo.startswith(base_nome) and arquivo.endswith(".csv"):
            try:
                numero = int(arquivo.replace(base_nome, "").replace(".csv", ""))
                numeros.append(numero)
            except:
                continue
    proximo_numero = max(numeros) + 1 if numeros else 1
    return f"{base_nome}{proximo_numero}.csv"

# Execução principal
if __name__ == "__main__":
    cores, rotulos, prop_harm, prop_nao_harm = gerar_cores(
        linhas=500000,
        proporcao_min_harmonicos=0.45,
        proporcao_min_nao_harmonicos=0.45
    )

    df = pd.DataFrame(cores, columns=["R", "G", "B"])
    df["RÓTULO"] = rotulos
    df["GRUPO"] = [i // 5 + 1 for i in range(len(df))]

    calcular_porcentagem_harmonia(rotulos)

    os.makedirs(diretorio_saida, exist_ok=True)
    nome_arquivo = determinar_nome_arquivo(diretorio_saida)
    caminho_saida = os.path.join(diretorio_saida, nome_arquivo)

    df.to_csv(caminho_saida, index=False)
    print(f"\nArquivo salvo em: {caminho_saida}")
