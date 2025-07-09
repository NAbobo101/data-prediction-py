import pandas as pd
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Define o caminho para a sua pasta de dados
# O script assume que a pasta 'database' está no mesmo diretório que o seu script Python
caminho_dados = 'database'

# Padrão para encontrar todos os arquivos de partidas da ATP
# Isso inclui os anuais, de qualificação e challengers
padrao_arquivos = os.path.join(caminho_dados, 'atp_matches*.csv')

# Usa a biblioteca glob para encontrar todos os arquivos que correspondem ao padrão
lista_arquivos = glob.glob(padrao_arquivos)

# Cria uma lista para armazenar cada DataFrame lido
lista_de_dataframes = []

print(f"Carregando {len(lista_arquivos)} arquivos de partidas...")

# Loop para ler cada arquivo e adicioná-lo à lista
for arquivo in lista_arquivos:
    try:
        df = pd.read_csv(arquivo, index_col=None, header=0)
        lista_de_dataframes.append(df)
        print(f"Arquivo '{arquivo}' carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao ler o arquivo {arquivo}: {e}")

# Concatena todos os DataFrames da lista em um único DataFrame
if lista_de_dataframes:
    dados_completos = pd.concat(lista_de_dataframes, axis=0, ignore_index=True)

    print("\nTodos os dados foram combinados!")
    print(f"Total de partidas carregadas: {len(dados_completos)}")

    # Exibe as primeiras 5 linhas para verificação
    print("\nAmostra dos dados:")
    print(dados_completos.head())
else:
    print("Nenhum arquivo de partida foi encontrado. Verifique o caminho e os nomes dos arquivos.")

print("\n--- Iniciando Passo 2: Limpeza e Engenharia de Features ---")

# 1. Seleção de Colunas Relevantes
# Baseado no que vimos, selecionamos colunas sobre os jogadores, o torneio e estatísticas.
colunas_relevantes = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_date',
    'winner_id', 'winner_name', 'loser_id', 'loser_name',
    'winner_ht', 'loser_ht', 'winner_age', 'loser_age',
    'score', 'best_of', 'round'
]
# Filtra o DataFrame para manter apenas as colunas relevantes
# Usamos .copy() para evitar avisos do Pandas
dados_limpos = dados_completos[colunas_relevantes].copy()

# 2. Tratamento de Dados Faltantes
# Removemos linhas onde informações cruciais (como idade ou altura) não estão presentes.
dados_limpos.dropna(subset=['winner_age', 'loser_age', 'winner_ht', 'loser_ht'], inplace=True)

# 3. Conversão e Ordenação por Data
# Convertemos a coluna 'tourney_date' para o formato datetime
dados_limpos['tourney_date'] = pd.to_datetime(dados_limpos['tourney_date'], format='%Y%m%d')
# Ordenamos todas as partidas da mais antiga para a mais nova. Isso é VITAL para o ELO.
dados_limpos.sort_values(by='tourney_date', inplace=True)

print("Dados limpos e ordenados cronologicamente.")

# 4. Implementação do Sistema de Classificação ELO
print("Calculando a classificação ELO para cada jogador...")

def atualizar_elo(elo_vencedor, elo_perdedor, k=32):
    """
    Calcula o novo ELO para o vencedor e o perdedor com base no resultado da partida.
    """
    prob_vencedor = 1 / (1 + 10 ** ((elo_perdedor - elo_vencedor) / 400))
    
    novo_elo_vencedor = elo_vencedor + k * (1 - prob_vencedor)
    novo_elo_perdedor = elo_perdedor - k * (1 - prob_vencedor)
    
    return novo_elo_vencedor, novo_elo_perdedor

# Dicionário para armazenar o ELO atual de cada jogador
elos_jogadores = {}
elo_inicial = 1500

# Listas para guardar os ELOs calculados antes de cada partida
elos_vencedor_partida = []
elos_perdedor_partida = []

# Iteramos sobre cada linha (partida) do DataFrame ordenado
for index, partida in dados_limpos.iterrows():
    vencedor = partida['winner_name']
    perdedor = partida['loser_name']
    
    # Busca o ELO atual do jogador ou define o ELO inicial se for a primeira vez que o vemos
    elo_vencedor_antes = elos_jogadores.get(vencedor, elo_inicial)
    elo_perdedor_antes = elos_jogadores.get(perdedor, elo_inicial)
    
    # Guarda o ELO dos jogadores ANTES da partida
    elos_vencedor_partida.append(elo_vencedor_antes)
    elos_perdedor_partida.append(elo_perdedor_antes)
    
    # Calcula o novo ELO DEPOIS da partida
    novo_elo_vencedor, novo_elo_perdedor = atualizar_elo(elo_vencedor_antes, elo_perdedor_antes)
    
    # Atualiza o ELO dos jogadores no nosso dicionário para a próxima partida deles
    elos_jogadores[vencedor] = novo_elo_vencedor
    elos_jogadores[perdedor] = novo_elo_perdedor

# Adiciona as novas colunas de ELO ao nosso DataFrame
dados_limpos['elo_vencedor'] = elos_vencedor_partida
dados_limpos['elo_perdedor'] = elos_perdedor_partida

# 5. Criação de Features de Diferença
# Essas são as features que o modelo de IA realmente usará para aprender
dados_limpos['diferenca_elo'] = dados_limpos['elo_vencedor'] - dados_limpos['elo_perdedor']
dados_limpos['diferenca_idade'] = dados_limpos['winner_age'] - dados_limpos['loser_age']
dados_limpos['diferenca_altura'] = dados_limpos['winner_ht'] - dados_limpos['loser_ht']

print("\nCálculo de ELO e criação de features de diferença concluídos!")
print("Amostra dos dados com as novas features:")
# Exibe as colunas mais importantes, incluindo as que acabamos de criar
print(dados_limpos[[
    'tourney_date', 'winner_name', 'loser_name', 
    'elo_vencedor', 'elo_perdedor', 'diferenca_elo', 
    'diferenca_idade', 'diferenca_altura'
]].tail())

print("\n--- Iniciando Passo 3: Treinamento dos Modelos de IA ---")

# 1. Preparação dos dados para o modelo
# Selecionamos as features que criamos (as diferenças) e o resultado
features = ['diferenca_elo', 'diferenca_idade', 'diferenca_altura']
X = dados_limpos[features]
# Criamos nosso alvo (y). Inicialmente, vamos considerar a vitória do 'winner' como 1
y = pd.Series([1] * len(dados_limpos))


# Para criar um modelo robusto, precisamos mostrar a ele os dois lados da moeda.
# Criamos um DataFrame invertido, onde o 'vencedor' se torna 'perdedor'.
# As diferenças são invertidas (ex: elo_vencedor - elo_perdedor se torna elo_perdedor - elo_vencedor)
X_invertido = -X
# O resultado para esses casos é 0.
y_invertido = pd.Series([0] * len(dados_limpos))


# Combinamos os dados originais e os invertidos
X_final = pd.concat([X, X_invertido], ignore_index=True)
y_final = pd.concat([y, y_invertido], ignore_index=True)


# 2. Divisão em Dados de Treino e Teste
# Usamos 80% dos dados para treinar o modelo e 20% para testar sua performance.
# O random_state=42 garante que a divisão seja sempre a mesma, para podermos reproduzir os resultados.
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")


# 3. Treinamento e Avaliação dos Modelos
print("\nTreinando e avaliando os modelos...")

# Modelo 1: Árvore de Decisão
print("  - Treinando Árvore de Decisão...")
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_train, y_train)
predicoes_arvore = modelo_arvore.predict(X_test)
acuracia_arvore = accuracy_score(y_test, predicoes_arvore)
print(f"    => Acurácia da Árvore de Decisão: {acuracia_arvore * 100:.2f}%")

# Modelo 2: Random Forest
print("  - Treinando Random Forest...")
modelo_floresta = RandomForestClassifier(random_state=42, n_estimators=100) # n_estimators é o número de árvores
modelo_floresta.fit(X_train, y_train)
predicoes_floresta = modelo_floresta.predict(X_test)
acuracia_floresta = accuracy_score(y_test, predicoes_floresta)
print(f"    => Acurácia do Random Forest: {acuracia_floresta * 100:.2f}%")


# Modelo 3: XGBoost
print("  - Treinando XGBoost...")
modelo_xgboost = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
modelo_xgboost.fit(X_train, y_train)
predicoes_xgboost = modelo_xgboost.predict(X_test)
acuracia_xgboost = accuracy_score(y_test, predicoes_xgboost)
print(f"    => Acurácia do XGBoost: {acuracia_xgboost * 100:.2f}%")