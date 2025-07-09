import pandas as pd
import glob
import os
from collections import deque
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

print("--- PROJETO DE PREVISÃO DE TÊNIS: VERSÃO DEFINITIVA ---")

# --- PASSO 1: CARREGAR DADOS DO CIRCUITO PRINCIPAL ---
caminho_dados = 'database'
# Padrão para pegar APENAS os arquivos do circuito principal (ex: atp_matches_2023.csv)
padrao_arquivos = os.path.join(caminho_dados, 'atp_matches_[0-9]*.csv')
lista_arquivos = glob.glob(padrao_arquivos)

if not lista_arquivos:
    print("ERRO: Nenhum arquivo do circuito principal (ex: atp_matches_2023.csv) foi encontrado na pasta 'database'.")
    print("Estes arquivos são essenciais para a alta acurácia. Por favor, baixe-os do repositório de Jeff Sackmann.")
else:
    print(f"Carregando {len(lista_arquivos)} arquivos do circuito principal...")
    dados_completos = pd.concat((pd.read_csv(f) for f in lista_arquivos), ignore_index=True)
    print(f"Total de partidas do circuito principal carregadas: {len(dados_completos)}")

    # --- PASSO 2: ENGENHARIA DE FEATURES AVANÇADA ---
    print("\n--- Iniciando Passo 2: Engenharia de Features com Estatísticas Detalhadas ---")

    # Colunas de estatísticas que usaremos
    stats_cols = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced']
    colunas_relevantes = [
        'tourney_date', 'surface', 'winner_name', 'loser_name', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age'
    ] + [f'w_{col}' for col in stats_cols] + [f'l_{col}' for col in stats_cols]

    dados_limpos = dados_completos[colunas_relevantes].copy()
    dados_limpos.dropna(inplace=True) # Remove qualquer partida sem stats completos
    dados_limpos = dados_limpos[dados_limpos['surface'].isin(['Clay', 'Grass', 'Hard'])]
    dados_limpos['tourney_date'] = pd.to_datetime(dados_limpos['tourney_date'], format='%Y%m%d')
    dados_limpos.sort_values(by='tourney_date', inplace=True)
    
    print(f"Total de partidas com estatísticas completas para análise: {len(dados_limpos)}")
    print("Calculando ELO e médias móveis de estatísticas...")

    # --- Dicionários para armazenar dados dinâmicos ---
    elo_inicial = 1500
    elos_por_superficie = {'Hard': {}, 'Clay': {}, 'Grass': {}}
    # Usaremos deque para performance, guardando as últimas 50 partidas de stats de um jogador
    stats_recentes = {}
    historico_vitorias = {} # Para % de vitórias

    features_para_modelo = []

    # --- Loop principal de engenharia de features ---
    for index, partida in dados_limpos.iterrows():
        superficie = partida['surface']
        vencedor, perdedor = partida['winner_name'], partida['loser_name']
        dict_elo_sup = elos_por_superficie[superficie]

        # 1. Pega ELO, Win% e Stats do VENCEDOR (antes da partida)
        elo_vencedor = dict_elo_sup.get(vencedor, elo_inicial)
        hist_vitorias_vencedor = historico_vitorias.setdefault(vencedor, deque(maxlen=50))
        win_pct_vencedor = sum(hist_vitorias_vencedor) / len(hist_vitorias_vencedor) if hist_vitorias_vencedor else 0.5
        stats_hist_vencedor = stats_recentes.setdefault(vencedor, deque(maxlen=50))
        avg_stats_vencedor = pd.DataFrame(stats_hist_vencedor).mean()

        # 2. Pega ELO, Win% e Stats do PERDEDOR (antes da partida)
        elo_perdedor = dict_elo_sup.get(perdedor, elo_inicial)
        hist_vitorias_perdedor = historico_vitorias.setdefault(perdedor, deque(maxlen=50))
        win_pct_perdedor = sum(hist_vitorias_perdedor) / len(hist_vitorias_perdedor) if hist_vitorias_perdedor else 0.5
        stats_hist_perdedor = stats_recentes.setdefault(perdedor, deque(maxlen=50))
        avg_stats_perdedor = pd.DataFrame(stats_hist_perdedor).mean()

        # 3. Cria as features de DIFERENÇA para o modelo
        partida_features = {
            'diferenca_elo': elo_vencedor - elo_perdedor,
            'diferenca_win_pct': win_pct_vencedor - win_pct_perdedor
        }
        
        for col in stats_cols:
            stat_vencedor = avg_stats_vencedor.get(col, 0)
            stat_perdedor = avg_stats_perdedor.get(col, 0)
            partida_features[f'diferenca_{col}'] = stat_vencedor - stat_perdedor
        
        features_para_modelo.append(partida_features)

        # 4. ATUALIZA os dados dos jogadores para o futuro
        # Atualiza ELO
        prob_vencedor = 1 / (1 + 10 ** ((elo_perdedor - elo_vencedor) / 400))
        novo_elo_vencedor = elo_vencedor + 32 * (1 - prob_vencedor)
        novo_elo_perdedor = elo_perdedor - 32 * (1 - prob_vencedor)
        dict_elo_sup[vencedor], dict_elo_sup[perdedor] = novo_elo_vencedor, novo_elo_perdedor
        
        # Atualiza históricos
        hist_vitorias_vencedor.append(1)
        hist_vitorias_perdedor.append(0)
        stats_hist_vencedor.append({col: partida[f'w_{col}'] for col in stats_cols})
        stats_hist_perdedor.append({col: partida[f'l_{col}'] for col in stats_cols})
    
    df_features = pd.DataFrame(features_para_modelo).fillna(0)

    # --- PASSO 3: TREINAMENTO E AVALIAÇÃO FINAL ---
    print("\n--- Iniciando Passo 3: Treinamento com Modelo Final ---")
    
    X = df_features
    y = pd.Series([1] * len(X))

    X_invertido = -X
    y_invertido = pd.Series([0] * len(X))

    X_final = pd.concat([X, X_invertido], ignore_index=True)
    y_final = pd.concat([y, y_invertido], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)
    print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    print("Treinando modelo XGBoost final...")
    modelo_final = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=1000, # Aumenta o número de árvores
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        random_state=42
    )
    
    # Treina com early stopping para evitar overfitting e achar o número ideal de árvores
    modelo_final.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_test, y_test)], verbose=False)
    
    predicoes = modelo_final.predict(X_test)
    acuracia = accuracy_score(y_test, predicoes)
    
    print("\n----------- RESULTADO FINAL -----------")
    print(f" => Acurácia final do modelo XGBoost: {acuracia * 100:.2f}%")
    print("---------------------------------------")

    if acuracia >= 0.75:
        print("\nOBJETIVO ATINGIDO! A combinação de features detalhadas foi o segredo.")
    elif acuracia > 0.65:
        print("\nResultado muito bom! Estamos no caminho certo e a qualidade dos dados foi fundamental.")
    else:
        print("\nO resultado melhorou. Para o próximo passo, poderíamos explorar a otimização de hiperparâmetros.")