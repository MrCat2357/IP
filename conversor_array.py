# Função para editar o arquivo e torná-lo compatível com o formato np.array
def formatar_matriz_em_array(arquivo_entrada, arquivo_saida):
    try:
        # Abrir o arquivo de entrada para leitura
        with open(arquivo_entrada, 'r') as f:
            # Ler todas as linhas do arquivo
            linhas = f.readlines()

        # Processar cada linha para garantir que está no formato correto
        matriz = []
        for linha in linhas:
            # Remover qualquer espaço extra e quebras de linha
            linha = linha.strip()
            
            # Separar os elementos por espaços e converter para lista
            elementos = linha.split()
            
            # Converter os elementos para inteiros (se for o caso)
            elementos = [int(x) for x in elementos]
            
            # Adicionar a linha formatada à matriz
            matriz.append(elementos)
        
        # Escrever no arquivo de saída no formato np.array
        with open(arquivo_saida, 'w') as f:
            f.write("np.array([\n")
            for i, linha in enumerate(matriz):
                f.write(f"    {linha}")
                if i < len(matriz) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("])\n")
        
        print("Arquivo formatado com sucesso!")
    
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

# Exemplo de uso:
arquivo_entrada = 'C:/Users/catul/OneDrive/Área de Trabalho/array.txt'  # Caminho do arquivo de entrada
arquivo_saida = 'C:/Users/catul/OneDrive/Área de Trabalho/saida.txt'  # Caminho do arquivo de saída

formatar_matriz_em_array(arquivo_entrada, arquivo_saida)
