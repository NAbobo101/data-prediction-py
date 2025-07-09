import pandas as pd
import glob
import os
import numpy as np

# Garante que estamos usando os dados ricos do circuito principal
caminho_dados = 'database'
padrao_arquivos = os.path.join(caminho_dados, 'atp_matches_[0-9]*.csv')
lista_arquivos = glob.glob(padrao_arquivos)

if not lista_arquivos:
    print("ERRO: Nenhum arquivo do circuito principal encontrado na pasta 'database'.")
else:
    print("Carregando dados para análise...")
    dados_completos = pd.concat((pd.read_csv(f, usecols=[
        'surface', 'winner_name', 'loser_name', 
        'w_ace', 'w_df', 'w_svpt', 'l_ace', 'l_df', 'l_svpt'
    ]) for f in lista_arquivos), ignore_index=True)
    
    # Limpa dados onde as estatísticas de saque não foram registradas
    dados_completos.dropna(subset=['w_svpt', 'l_svpt'], inplace=True)
    dados_completos = dados_completos[dados_completos['w_svpt'] > 0]
    dados_completos = dados_completos[dados_completos['l_svpt'] > 0]
    
    print("\n--- Análise de Padrões: Desempenho de Roger Federer por Superfície ---")
    
    # Filtra todas as partidas do jogador escolhido
    nome_jogador = 'Roger Federer'
    partidas_jogador = dados_completos[(dados_completos['winner_name'] == nome_jogador) | 
                                       (dados_completos['loser_name'] == nome_jogador)].copy()
    
    # --- Engenharia de Features para Análise ---
    # Cria uma coluna 'vitoria' (1 se ganhou, 0 se perdeu)
    partidas_jogador['vitoria'] = np.where(partidas_jogador['winner_name'] == nome_jogador, 1, 0)
    
    # Calcula as estatísticas de saque DO JOGADOR em cada partida
    partidas_jogador['ace_pct'] = np.where(
        partidas_jogador['winner_name'] == nome_jogador,
        partidas_jogador['w_ace'] / partidas_jogador['w_svpt'],
        partidas_jogador['l_ace'] / partidas_jogador['l_svpt']
    )
    partidas_jogador['df_pct'] = np.where(
        partidas_jogador['winner_name'] == nome_jogador,
        partidas_jogador['w_df'] / partidas_jogador['w_svpt'],
        partidas_jogador['l_df'] / partidas_jogador['l_svpt']
    )
    
    # --- Agrupamento e Análise de Padrão ---
    # Agrupa por superfície e calcula a média das nossas métricas
    analise_superficie = partidas_jogador.groupby('surface').agg(
        total_partidas=('vitoria', 'count'),
        percentual_vitorias=('vitoria', 'mean'),
        media_ace_pct=('ace_pct', 'mean'),
        media_df_pct=('df_pct', 'mean')
    ).reset_index()
    
    # Formata os resultados para melhor visualização
    analise_superficie['percentual_vitorias'] = (analise_superficie['percentual_vitorias'] * 100).map('{:.2f}%'.format)
    analise_superficie['media_ace_pct'] = (analise_superficie['media_ace_pct'] * 100).map('{:.2f}%'.format)
    analise_superficie['media_df_pct'] = (analise_superficie['media_df_pct'] * 100).map('{:.2f}%'.format)

    print(analise_superficie)