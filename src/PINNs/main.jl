# 
# Algoritmo para cálculo de rede neural
#
using MKL

using StatsFuns # função sigmoide (logística)
using StatsBase # Função Sample
using LinearAlgebra # cálculo de norma
using Plots # Gráficos
using Random; Random.seed!(1234) # define um seed para as variáveis aleatórias 
using Enzyme # diferenciação automática
using ProgressMeter # Barra de progresso ao rodar o código
using DelimitedFiles

# Adiciona demais arquivos do programa
include("struct_rede.jl")
include("derivadas.jl")
include("dados_treino.jl")
include("objetivo.jl")
include("RNA.jl")
include("adam.jl")
include("ativ.jl")
include("acuracia.jl")
include("perdas.jl")

# Função principal do código
function main(topologia::Vector{Int64}, ativ::Tuple, m::Float64, δ::Float64, ω0::Float64)

    # Cria a rede
    rede = Rede(topologia, ativ)

    # Inicializa os dados de treino
    treino = Treino(m, δ, ω0)

    # Chama a rotina de otimização do Adam
    x, objetivo_treino = Adam(rede, treino)

    # Atualiza pesos e bias com o resultado da otimização
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Agora vamos calcular a resposta em cada tempo 
    u_test_pred = zeros(1, size(treino.u_an,2))

    # Obtém a resposta da rede neural para os pontos de teste
    for i=1:size(treino.u_an,2)
        t = treino.t_teste[1,i]
        u_test_pred[:, i] = RNA(rede, pesos, bias, [t])
    end

    # Retorna as variáveis de projeto, função objetivo ao longo do tempo,
    # resposta analítica nos pontos de teste e resposta calculada pela rede neural
    return x, objetivo_treino, treino.u_an, u_test_pred

end

# Roda a rede neural
function roda()

    # Define os dados do problema: topologia e funções de ativação
    topologia = [1; 100; 50; 1]
    #ativ = (ReLU, ReLU, ReLU, identity)
    ativ = (tanh, tanh, tanh)

    # Parâmetros do sistema
    m = 1.0
    δ = 2.0
    ω0 = 20.0

    # Roda a função main
    x, objetivo_treino, u_an, u_test_pred = main(topologia, ativ, m, δ, ω0)

    # Acompanha a evolução do objetivo ao longo do tempo
    display(plot([objetivo_treino], title = "Objetivo", label = ["Treino"]))

    # Compara a resposta analítica com a calculada pela rede neural
    display(plot([u_an', u_test_pred'], title = "Deslocamento", label = ["Analítico" "Rede neural"]))

        #display(plot([u_test_pred'], title = "Deslocamento", label = ["Rede neural"]))


   return x, objetivo_treino, u_an, u_test_pred

end