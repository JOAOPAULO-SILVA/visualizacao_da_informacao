import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
df_msg = pd.read_csv('data_messages.csv')

# 2. Filtrar apenas mensagens diretas (one-on-one)
df_1v1 = df_msg[df_msg['message_type'] == 'one_on_one_chat'].copy()

# 3. O Pulo do Gato: Cruzar quem respondeu com quem enviou a original
# Criamos um mini-dataframe só com os IDs das mensagens e quem enviou
df_autores = df_msg[['message_id', 'agent_role']].rename(columns={'agent_role': 'target_role'})

# Juntamos usando o 'responding_to'. 
# Se o Agente A respondeu a mensagem X do Agente B, criamos uma conexão A -> B.
df_conexoes = df_1v1.merge(df_autores, left_on='responding_to', right_on='message_id', how='inner')

# 4. Contar a frequência das conversas entre cada par de agentes
interacoes = df_conexoes.groupby(['agent_role', 'target_role']).size().reset_index(name='quantidade')

# ==========================================
# VISUALIZAÇÃO 1: GRAFO DE REDE (NETWORKX)
# ==========================================
plt.figure(figsize=(10, 6))
plt.title('Grafo de Rede: Interações Diretas (One-on-One)', fontsize=14, fontweight='bold')

# Cria um grafo direcionado (mostra quem iniciou a resposta)
G = nx.from_pandas_edgelist(interacoes, 
                            source='agent_role', 
                            target='target_role', 
                            edge_attr='quantidade', 
                            create_using=nx.DiGraph())

# Layout do grafo (circular fica ótimo para poucos agentes)
pos = nx.circular_layout(G)

# Desenhar nós
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='skyblue', edgecolors='black')

# Desenhar arestas (linhas) com espessura baseada na quantidade de mensagens
pesos = [G[u][v]['quantidade'] * 0.5 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=pesos, edge_color='gray', connectionstyle='arc3,rad=0.1')

# Adicionar os nomes
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.axis('off')
plt.savefig("teste1.svg")


# ==========================================
# VISUALIZAÇÃO 2: HEATMAP (A MINHA RECOMENDADA)
# ==========================================
plt.figure(figsize=(8, 6))
plt.title('Mapa de Calor: Frequência de Mensagens Diretas', fontsize=14, fontweight='bold')

# Transforma a lista de interações em uma matriz (linhas=remetente, colunas=destinatário)
matriz_interacao = interacoes.pivot(index='agent_role', columns='target_role', values='quantidade').fillna(0)

# Desenha o heatmap
sns.heatmap(matriz_interacao, annot=True, fmt=".0f", cmap="Blues", linewidths=.5, cbar_kws={'label': 'Nº de Mensagens'})

plt.ylabel('Quem Respondeu')
plt.xlabel('Quem Recebeu a Resposta')
plt.tight_layout()
plt.savefig("teste2.svg")