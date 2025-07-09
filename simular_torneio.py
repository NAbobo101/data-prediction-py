import pandas as pd
import glob
import os
from collections import deque
from xgboost import XGBClassifier
import numpy as np
import warnings

# Ignora avisos para uma saída mais limpa
warnings.simplefilter(action='ignore', category=FutureWarning)

def simular_torneio_wimbledon():
    """
    Script completo que treina o modelo especialista e o utiliza para
    simular o restante do torneio de Wimbledon 2025 e prever o campeão.
    (Versão compatível com bibliotecas mais antigas de XGBoost)
    """
    print("--- SIMULADOR DO TORNEIO DE WIMBLEDON 2025 (VERSÃO COMPATÍVEL) ---")

    # --- PASSO 1: CARREGAR E PREPARAR OS DADOS ---
    print("\n--- Carregando e Preparando os Dados Históricos ---")
    caminho_dados = 'database'
    padrao_arquivos = os.path.join(caminho_dados, 'atp_matches_[0-9]*.csv')
    lista_arquivos = glob.glob(padrao_arquivos)

    if not lista_arquivos:
        print("ERRO: Nenhum arquivo do circuito principal (ex: atp_matches_2023.csv) foi encontrado.")
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
    print("\n--- Calculando a 'forma' atual de todos os jogadores... ---")
    historico_features = {} 
    feature_cols_ordered = ['win_pct', 'ace_pct', 'df_pct', '1st_serve_win_pct', 'bp_saved_pct']

    for index, partida in dados_limpos.iterrows():
        vencedor, perdedor = partida['winner_name'], partida['loser_name']
        hist_vencedor = historico_features.setdefault(vencedor, deque(maxlen=50))
        hist_perdedor = historico_features.setdefault(perdedor, deque(maxlen=50))
        
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
    
    # --- PASSO 3: TREINAMENTO DO MODELO ESPECIALISTA ---
    print("\n--- Treinando o Modelo Especialista... (Isso pode levar alguns minutos) ---")
    
    features_para_modelo = []
    for index, partida in dados_limpos.iterrows():
        vencedor, perdedor = partida['winner_name'], partida['loser_name']
        # Usamos .get() para evitar erros se um jogador não tiver histórico ainda
        avg_features_vencedor = pd.DataFrame(historico_features.get(vencedor)).mean()
        avg_features_perdedor = pd.DataFrame(historico_features.get(perdedor)).mean()
        partida_features = {}
        for col in feature_cols_ordered:
            partida_features[f'diff_{col}'] = avg_features_vencedor.get(col, 0.5 if col == 'win_pct' else 0) - avg_features_perdedor.get(col, 0.5 if col == 'win_pct' else 0)
        features_para_modelo.append(partida_features)

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
        use_label_encoder=False, random_state=42
    )
    
    # ****** INÍCIO DA CORREÇÃO ******
    # Chamada ao .fit() simplificada para remover os parâmetros incompatíveis
    modelo_final.fit(X_final, y_final, verbose=False)
    # ****** FIM DA CORREÇÃO ******
    
    print("Modelo treinado com sucesso!")
    
    # --- PASSO 4: SIMULAÇÃO DO TORNEIO ---
    
    def prever_vencedor(jogador_a, jogador_b):
        hist_a = historico_features.get(jogador_a)
        hist_b = historico_features.get(jogador_b)

        if not hist_a or not hist_b:
            vencedor_wo = jogador_a if hist_a else (jogador_b if hist_b else jogador_a)
            print(f"  Aviso: {vencedor_wo} avança por W.O. (sem dados históricos para o oponente)")
            return vencedor_wo

        forma_a = pd.DataFrame(hist_a).mean()
        forma_b = pd.DataFrame(hist_b).mean()
        
        features_previsao = []
        for col in feature_cols_ordered:
            diff = forma_a.get(col, 0.5 if col == 'win_pct' else 0) - forma_b.get(col, 0.5 if col == 'win_pct' else 0)
            features_previsao.append(diff)
            
        dados_partida = np.array([features_previsao])
        prob_a = modelo_final.predict_proba(dados_partida)[0][1]
        
        vencedor = jogador_a if prob_a > 0.5 else jogador_b
        print(f"  Confronto: {jogador_a} vs {jogador_b} => Vencedor Previsto: {vencedor} ({max(prob_a, 1-prob_a)*100:.1f}%)")
        return vencedor

    def simular_rodada(nome_rodada, confrontos):
        print(f"\n--- Simulação: {nome_rodada} ---")
        vencedores = []
        for j1, j2 in confrontos:
            vencedor = prever_vencedor(j1, j2)
            vencedores.append(vencedor)
        return vencedores

    quartas_de_final_wimbledon = [
        ("Jannik Sinner", "Ben Shelton"),
        ("Flavio Cobolli", "Novak Djokovic"),
        ("Taylor Fritz", "Karen Khachanov"),
        ("Cameron Norrie", "Carlos Alcaraz")
    ]
    
    vencedores_quartas = simular_rodada("Quartas de Final", quartas_de_final_wimbledon)
    semifinais = [(vencedores_quartas[i], vencedores_quartas[i+1]) for i in range(0, len(vencedores_quartas), 2)]
    vencedores_semifinais = simular_rodada("Semifinais", semifinais)
    final = [(vencedores_semifinais[0], vencedores_semifinais[1])]
    campeao = simular_rodada("FINAL", final)

    print("\n" + "="*50)
    print(f"🏆 CAMPEÃO PREVISTO DE WIMBLEDON 2025: {campeao[0]} 🏆")
    print("="*50)


# Bloco de execução principal
if __name__ == '__main__':
    simular_torneio_wimbledon()