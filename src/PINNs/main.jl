# 
# Algoritmo para cálculo de rede neural
#

using MKL
using StatsFuns # função sigmoide (logística)
using StatsBase # Função Sample
using LinearAlgebra # cálculo de norma
using Plots # Gráficos
using Random; # Random.seed!(1234) # define um seed para as variáveis aleatórias 
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
include("resultados.jl")
include("perdas.jl")

# Função principal do código
function main(topologia::Vector{Int64}, ativ::Tuple, m::Float64, δ::Float64, ω0::Float64,
              nepoch::Int64)

    # Cria a rede
    rede = Rede(topologia, ativ)

    # Inicializa os dados de treino
    treino = Treino(m, δ, ω0)

    # Chama a rotina de otimização do Adam
    x, objetivo_treino, u_test_pred = Adam(rede, treino, nepoch)

    # Retorna as variáveis de projeto, função objetivo ao longo do tempo,
    # resposta analítica nos pontos de teste e resposta calculada pela rede neural
    return x, objetivo_treino, treino, u_test_pred, rede

end

# Roda a rede neural
function roda()

    # Define os dados do problema: topologia e funções de ativação
    topologia = [1; 100; 50; 50; 50; 1]
    ativ = (tanh, tanh, tanh, tanh, tanh)

    # Número de épocas
    nepoch = 15_000

    # Parâmetros do sistema
    m = 1.0
    δ = 2.0
    ω0 = 20.0

    # Roda a função main
    x, objetivo_treino, treino, u_test_pred = main(topologia, ativ, m, δ, ω0, nepoch)

   return x, objetivo_treino, treino, u_test_pred, rede

end

#
# Rotina para brincar com a rede, dado um vetor x 
#
function Brincando(arquivo_x)

   # Le os pesos 
   x = readdlm(arquivo_x)

   # Define os dados do problema: topologia e funções de ativação
   topologia = [1; 100; 50; 50; 50; 1]
   ativ = (tanh, tanh, tanh, tanh, tanh)
   
   # Parâmetros do sistema
   m = 1.0
   δ = 2.0
   ω0 = 20.0

   # Cria a rede
   rede = Rede(topologia, ativ)

   # Inicializa os dados de treino
   treino = Treino(m, δ, ω0)

   return rede, treino, x

end