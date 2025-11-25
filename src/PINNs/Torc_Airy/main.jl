# 
# Algoritmo para cálculo de rede neural
#

using MKL # Cálculo matricial
using StatsFuns # função sigmoide (logística)
using StatsBase # Função Sample
using LinearAlgebra # cálculo de norma
using Plots # Gráficos
using Random; # Random.seed!(1234) # define um seed para as variáveis aleatórias 
using Enzyme # diferenciação automática
using ProgressMeter # Barra de progresso ao rodar o código
using DelimitedFiles # Escrever e ler arquivos

# Adiciona demais arquivos do programa
include("struct_rede.jl")
include("derivadas.jl")
include("dados_treino.jl")
include("objetivo.jl")
include("RNA.jl")
include("adamW.jl")
include("ativ.jl")
include("resultados.jl")
include("perdas.jl")
include("geometria.jl")
include("inicial.jl")
include("eq_diff.jl")
#include("LBFGS.jl")
#include("line_search.jl")

# Função principal do código
function main(topologia::Vector{Int64}, ativ::Tuple, nepoch_ADAM::Int64, nepoch_LBFGS::Int64, prob::String)

    # Cria a rede
    rede = Rede(topologia, ativ)

    # Inicializa os dados de treino
    dict_treino = Treino(prob)

    # Chama a rotina de otimização do AdamW
    x, objetivo_treino, u_test_pred = AdamW(rede, dict_treino, nepoch_ADAM)

    # Chama a rotina de otimização do LBFGS
    #x, objetivo_treino, u_test_pred = LBFGS(rede, treino, nepoch_LBFGS)

    # Retorna as variáveis de projeto, função objetivo ao longo do tempo,
    # resposta analítica nos pontos de teste e resposta calculada pela rede neural
    return x, objetivo_treino, treino, u_test_pred, rede
    
end

# Roda a rede neural
function roda()

   # Define os dados do problema: topologia e funções de ativação
   topologia = [1; 50; 50; 50; 1]
   ativ = (tanh, tanh, tanh, identity)

   # Número de épocas
   nepoch_ADAM = 15_000
   nepoch_LBFGS = 3_000

   # Problema a ser resolvido
   prob = "circular"

   # Roda a função main
   x, objetivo_treino, treino, u_test_pred, rede = main(topologia, ativ, nepoch_ADAM, nepoch_LBFGS, prob)
   
   # Retorna variáveis de projeto e outras informações   
   return x, objetivo_treino, treino, u_test_pred, rede
   
end