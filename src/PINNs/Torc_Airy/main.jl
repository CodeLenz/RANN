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
using StaticArrays # Tipos de vetores estáticos e mutáveis

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
include("circular.jl")
include("retangular.jl")
include("sec_L.jl")
include("dist_sukumar.jl")
include("inicial.jl")
include("eq_diff.jl")
include("LBFGS.jl") 

# Função principal do código
function main(topologia::Vector{Int64}, ativ::Tuple, nepoch_ADAM::Int64, nepoch_LBFGS::Int64, prob::String)

    # Cria a rede
    rede = Rede(topologia, ativ)

    # Inicializa os dados de treino
    treino = Treino(prob)

    println("***********************")
    println("EXPLORANDO COM O ADAMW")
    println("***********************")

    # Chama a rotina de otimização do AdamW
    x, objetivo_treino_adam, u_test_pred_adam = AdamW(rede, treino, nepoch_ADAM, prob)

    # Atualiza os pesos e bias da rede com os resultados otimizados do AdamW
    # para que o L-BFGS continue de onde o AdamW parou.
    rede.x .= x 

    println("***********************")
    println("PASSANDO PARA O L-BFGS")
    println("***********************")

    # Chama a rotina de otimização do LBFGS
    x, objetivo_treino_lbfgs, u_test_pred = LBFGS(rede, treino, nepoch_LBFGS, prob)

    # Concatena os históricos de objetivo e gera gráfico completo
    objetivo_treino_total = vcat(objetivo_treino_adam, objetivo_treino_lbfgs)
    plot_obj_total = plot([objetivo_treino_total], title = "Objetivo ADAM + LBFGS", label = ["Treino"], size = (1000, 1000))
    savefig(plot_obj_total, "Resultados/plot_obj_total.png")

    # Retorna as variáveis de projeto, função objetivo ao longo do tempo,
    # resposta analítica nos pontos de teste e resposta calculada pela rede neural
    return x, objetivo_treino_total, treino, u_test_pred, rede
    
end

# Roda a rede neural
function roda()

   # Define os dados do problema: topologia e funções de ativação
   topologia = [2; 50; 50; 50; 1]
   ativ = (tanh, tanh, tanh, identity)

   # Número de épocas
   nepoch_ADAM = 5_000
   nepoch_LBFGS = 5_000

   # Problema a ser resolvido
   # prob = "Circular"
   # prob = "Retangular"
   prob = "L"

   # Roda a função main
   x, objetivo_treino, treino, u_test_pred, rede = main(topologia, ativ, nepoch_ADAM, nepoch_LBFGS, prob)
   
   # Retorna variáveis de projeto e outras informações   
   return x, objetivo_treino, treino, u_test_pred, rede
   
end