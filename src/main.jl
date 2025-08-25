# 
# Algoritmo para cálculo de rede neural
#

# Bibliotecas  
using StatsFuns # função sigmoide (logística)
using StatsBase # Função Sample
using LinearAlgebra # cálculo de norma
using Plots # Gráficos
using Random; Random.seed!(1234) # define um seed para as variáveis aleatórias 
using Enzyme # diferenciação automática
using DelimitedFiles # Leitura dos arquivos .csv
using CodecZlib # Leitura de arquivos comprimidos (.gz)
using ProgressMeter # Barra de progresso ao rodar o código

# Adiciona demais arquivos do programa
include("struct_rede.jl")
include("dados_treino.jl")
include("objetivo.jl")
include("RNA.jl")
include("adam.jl")
include("ativ.jl")
include("acuracia.jl")

# Função principal do código
function main(topologia::Vector{Int64}, ativ::Tuple)

    # Cria a rede
    rede = Rede(topologia, ativ)

    # Inicializa os dados de treino
    treino = Treino(topologia)

    # Incializa x
    x = randn(rede.n_projeto)

    # Chama a rotina de otimização do Adam
    x, objetivo_treino, objetivo_teste = Adam(rede, treino)

    # Atualiza pesos e bias com o resultado da otimização
    pesos, bias = Atualiza_pesos_bias(rede, x)
    
    # Calcula a acurácia do modelo e vetor de labels de teste estimado
    acuracia_modelo, labels_teste_calc = acuracia(rede, treino, pesos, bias)

    # Retorna as variáveis de projeto, função objetivo ao longo do tempo,
    # acurácia do modelo e as labels esperadas e estimadas pela rede neural
    return x, objetivo_treino, objetivo_teste, acuracia_modelo, labels_teste_calc, treino.labels_teste

end

# Roda a rede neural
function roda()

    # Define os dados do problema: topologia e funções de ativação
    # A última função deve ser a identidade (i.e. não altera os logits),
    # pois a função de ativação softmax já está embutida dentro da função objetivo
    topologia = [784; 30; 30; 10]
    ativ = (ReLU, ReLU, identity)

    # Roda a função main
    x, objetivo_treino, objetivo_teste, acuracia, labels_teste_calc, labels_teste = main(topologia, ativ)

    # Acompanha a evolução do objetivo ao longo do tempo
    display(plot([objetivo_treino, objetivo_teste], title = "Objetivo", label = ["Treino" "Teste"]))

    # Mostra a acurácia do modelo
    print("A acurácia do modelo é de $acuracia%")

end