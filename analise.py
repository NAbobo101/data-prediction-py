import pandas as pd
import glob
import os
import numpy as np

def gerar_analise_em_massa():
    """
    Carrega todos os dados do circuito principal, processa as estatísticas
    para todos os jogadores e salva o resultado em um arquivo CSV.
    """
    print("--- Iniciando Análise em Massa e Geração de Features ---")

    # --- Parte 1: Carregamento dos Dados ---
    caminho_dados = 'database'
    padrao_arquivos = os.path.join(caminho_dados, 'atp_matches_[0-9]*.csv')
    lista_arquivos = glob.glob(padrao_arquivos)

    if not lista_arquivos:
        print(f"ERRO: Nenhum arquivo do circuito principal encontrado na pasta '{caminho_dados}'.")
        return

    print(f"Carregando {len(lista_arquivos)} arquivos do circuito principal...")
    colunas_necessarias = [
        'surface', 'winner_name', 'loser_name', 
        'w_ace', 'w_df', 'w_svpt', 'l_ace', 'l_df', 'l_svpt'
    ]
    dados_completos = pd.concat(
        (pd.read_csv(f, usecols=lambda col: col in colunas_necessarias, on_bad_lines='skip') for f in lista_arquivos),
        ignore_index=True
    )
    
    # --- Parte 2: Limpeza ---
    dados_completos.dropna(subset=['w_svpt', 'l_svpt', 'surface'], inplace=True)
    dados_completos = dados_completos[dados_completos['w_svpt'] > 0]
    dados_completos = dados_completos[dados_completos['l_svpt'] > 0]
    print("Limpeza de dados concluída.")

    # --- Parte 3: Reestruturação dos Dados (Unpivot) ---
    print("Reestruturando dados para análise por jogador...")
    # Cria um DataFrame para os vencedores
    vencedores = dados_completos.copy()
    vencedores['player_name'] = vencedores['winner_name']
    vencedores['vitoria'] = 1
    vencedores['ace'] = vencedores['w_ace']
    vencedores['df'] = vencedores['w_df']
    vencedores['svpt'] = vencedores['w_svpt']

    # Cria um DataFrame para os perdedores
    perdedores = dados_completos.copy()
    perdedores['player_name'] = perdedores['loser_name']
    perdedores['vitoria'] = 0
    perdedores['ace'] = perdedores['l_ace']
    perdedores['df'] = perdedores['l_df']
    perdedores['svpt'] = perdedores['l_svpt']

    # Combina os dois DataFrames
    dados_por_jogador = pd.concat([vencedores, perdedores], ignore_index=True)
    
    # Seleciona apenas as colunas que importam para a análise final
    dados_por_jogador = dados_por_jogador[['player_name', 'surface', 'vitoria', 'ace', 'df', 'svpt']]

    # Calcula as taxas de ace e dupla falta
    dados_por_jogador['ace_pct'] = dados_por_jogador['ace'] / dados_por_jogador['svpt']
    dados_por_jogador['df_pct'] = dados_por_jogador['df'] / dados_por_jogador['svpt']
    
    # --- Parte 4: Agrupamento em Massa ---
    print("Agrupando dados para calcular estatísticas por jogador e superfície...")
    analise_final = dados_por_jogador.groupby(['player_name', 'surface']).agg(
        total_partidas=('vitoria', 'count'),
        percentual_vitorias=('vitoria', 'mean'),
        media_ace_pct=('ace_pct', 'mean'),
        media_df_pct=('df_pct', 'mean')
    ).reset_index()

    # Filtra jogadores com um número mínimo de partidas para relevância estatística
    analise_final = analise_final[analise_final['total_partidas'] >= 20]

    # --- Parte 5: Salvando o Resultado ---
    nome_arquivo_saida = 'analise_geral_jogadores.csv'
    analise_final.to_csv(nome_arquivo_saida, index=False)
    
    print("\n" + "="*60)
    print(f"ANÁLISE EM MASSA CONCLUÍDA!")
    print(f"Os dados foram salvos no arquivo: '{nome_arquivo_saida}'")
    print("="*60)
    print("Amostra do resultado final:")
    print(analise_final.sort_values(by='total_partidas', ascending=False).head(10))
    print("="*60)


if __name__ == '__main__':
    gerar_analise_em_massa()