import pandas as pd
import glob
import os
from collections import deque
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def prever_nova_partida():
    """
    Script completo que treina o modelo especialista e o utiliza para
    prever o resultado de uma nova partida específica.
    """
    print("--- PROJETO DE PREVISÃO DE TÊNIS: TREINAMENTO E PREVISÃO ---")

    # --- PASSO 1: CARREGAR E PREPARAR OS DADOS ---
    print("\n--- Carregando e Preparando os Dados ---")
    caminho_dados = 'database'
    padrao_arquivos = os.path.join(caminho_dados, 'atp_matches_[0-9]*.csv')
    lista_arquivos = glob.glob(padrao_arquivos)

    if not lista_arquivos:
        print("ERRO: Nenhum arquivo do circuito principal foi encontrado.")
        return

    dados_completos = pd.concat((pd.read_csv(f, on_bad_lines='skip') for f in lista_arquivos), ignore_index=True)

    stats_cols = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced']
    colunas_relevantes = ['tourney_date', 'surface', 'winner_name', 'loser_name'] + [f'w_{col}' for col in stats_cols] + [f'l_{col}' for col in stats_cols]
    
    dados_limpos = dados_completos[colunas_relevantes].copy()
    dados_limpos.dropna(inplace=True)
    dados_limpos = dados_limpos[dados_limpos['w_svpt'] > 0]
    dados_limpos = dados_limpos[dados_limpos['l_svpt'] > 0]
    dados_limpos.sort_values(by='tourney_date', inplace=True)
    
    # --- PASSO 2: ENGENHARIA DE FEATURES COM MÉDIAS MÓVEIS ---
    print("\n--- Calculando a 'forma' atual dos jogadores (Médias Móveis)... ---")
    historico_features = {} 
    features_para_modelo = []

    # Este loop calcula as features para o treinamento
    for index, partida in dados_limpos.iterrows():
        vencedor, perdedor = partida['winner_name'], partida['loser_name']
        hist_vencedor = historico_features.setdefault(vencedor, deque(maxlen=50))
        hist_perdedor = historico_features.setdefault(perdedor, deque(maxlen=50))

        avg_features_vencedor = pd.DataFrame(hist_vencedor).mean()
        avg_features_perdedor = pd.DataFrame(hist_perdedor).mean()

        partida_features = {}
        # As colunas precisam ser criadas na mesma ordem para a previsão
        feature_cols_ordered = ['win_pct', 'ace_pct', 'df_pct', '1st_serve_win_pct', 'bp_saved_pct']
        for col in feature_cols_ordered:
             partida_features[f'diff_{col}'] = avg_features_vencedor.get(col, 0.5 if col == 'win_pct' else 0) - avg_features_perdedor.get(col, 0.5 if col == 'win_pct' else 0)
        features_para_modelo.append(partida_features)

        # Atualiza o histórico com as taxas de eficiência desta partida
        w_stats = {
            'win_pct': 1,
            'ace_pct': partida['w_ace'] / partida['w_svpt'],
            'df_pct': partida['w_df'] / partida['w_svpt'],
            '1st_serve_win_pct': partida['w_1stWon'] / partida['w_1stIn'] if partida['w_1stIn'] > 0 else 0,
            'bp_saved_pct': partida['w_bpSaved'] / partida['w_bpFaced'] if partida['w_bpFaced'] > 0 else 0
        }
        hist_vencedor.append(w_stats)
        l_stats = {
            'win_pct': 0,
            'ace_pct': partida['l_ace'] / partida['l_svpt'],
            'df_pct': partida['l_df'] / partida['l_svpt'],
            '1st_serve_win_pct': partida['l_1stWon'] / partida['l_1stIn'] if partida['l_1stIn'] > 0 else 0,
            'bp_saved_pct': partida['l_bpSaved'] / partida['l_bpFaced'] if partida['l_bpFaced'] > 0 else 0
        }
        hist_perdedor.append(l_stats)
    
    # --- PASSO 3: TREINAMENTO DO MODELO ---
    print("\n--- Treinando o Modelo Especialista... ---")
    df_features = pd.DataFrame(features_para_modelo).fillna(0)
    X = df_features
    y = pd.Series([1] * len(X))
    X_invertido = -X
    y_invertido = pd.Series([0] * len(X))
    X_final = pd.concat([X, X_invertido], ignore_index=True)
    y_final = pd.concat([y, y_invertido], ignore_index=True)

    modelo_final = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, random_state=42
    )
    
    modelo_final.fit(X_final, y_final, verbose=False)
    print("Modelo treinado com sucesso!")
    
    # --- PASSO 4: PREVISÃO DA PARTIDA ---
    print("\n" + "="*50)
    print("FAZENDO A PREVISÃO DA PARTIDA")
    print("="*50)
    
    jogador_a_nome = "Taro Daniel"
    jogador_b_nome = "Jenson Brooksby"

    # Pega o histórico mais recente dos jogadores, que foi calculado no loop acima
    hist_jogador_a = historico_features.get(jogador_a_nome)
    hist_jogador_b = historico_features.get(jogador_b_nome)

    if not hist_jogador_a or not hist_jogador_b:
        print(f"ERRO: Não foi possível encontrar o histórico de um dos jogadores: {jogador_a_nome} ou {jogador_b_nome}")
        return

    # Calcula a forma atual (média das últimas 50 partidas)
    forma_jogador_a = pd.DataFrame(hist_jogador_a).mean()
    forma_jogador_b = pd.DataFrame(hist_jogador_b).mean()
    
    print(f"Forma recente de {jogador_a_nome}:\n{forma_jogador_a.to_string()}\n")
    print(f"Forma recente de {jogador_b_nome}:\n{forma_jogador_b.to_string()}\n")

    # Cria o vetor de features para a previsão, na mesma ordem do treinamento
    features_previsao = []
    for col in feature_cols_ordered:
        diff = forma_jogador_a.get(col, 0.5 if col == 'win_pct' else 0) - forma_jogador_b.get(col, 0.5 if col == 'win_pct' else 0)
        features_previsao.append(diff)
        
    dados_partida = np.array([features_previsao])

    # Faz a previsão de probabilidade
    probabilidade_vitoria_a = modelo_final.predict_proba(dados_partida)[0][1]
    probabilidade_vitoria_b = 1 - probabilidade_vitoria_a

    print("--- RESULTADO DA PREVISÃO ---")
    print(f"Probabilidade de vitória para {jogador_a_nome}: {probabilidade_vitoria_a * 100:.2f}%")
    print(f"Probabilidade de vitória para {jogador_b_nome}: {probabilidade_vitoria_b * 100:.2f}%")

    vencedor_provavel = jogador_a_nome if probabilidade_vitoria_a > 0.5 else jogador_b_nome
    print(f"\n=> O modelo prevê que o vencedor será: {vencedor_provavel}")
    print("="*50)


# Bloco de execução principal
if __name__ == '__main__':
    prever_nova_partida()