module RANN

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
include("main.jl")

# Exporta as rotinas 
export roda

end
