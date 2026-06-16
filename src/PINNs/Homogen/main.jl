# =============================================================================
#  Programa de homogeneização usando PINNs
#  O programa minimiza a energia de deformação da célula unitária utilizando
#  integração Quasi-Monte Carlo (Sobol).
#
#  Difereciação automática feita em código + pullback do Zygote para cálculo da
#  derivada da função de perda em relação aos parâmetros do modelo.
#
# =============================================================================
using LinearAlgebra
using Zygote
using Sobol
using Plots
using DelimitedFiles
using ProgressMeter

include("ativ.jl")
include("RNA.jl")
include("PINN.jl")
include("material.jl")
include("tensao_pos.jl")
include("treino.jl")
include("AdamW.jl")
include("L-BFGS/L-BFGS.jl")
include("L-BFGS/dois_lacos.jl")
include("L-BFGS/refinamento.jl")
include("L-BFGS/wolfe_ls.jl")
include("L-BFGS/interpolacao.jl")

# =============================================================================
#  Validação
# =============================================================================
function Main_Homogenizacao()

    # Propriedades dos materiais da célula
    # no caso, fibra e matriz
    # raio e centro da fibra
    mat_params = (
        E_m = 1.0, ν_m = 0.3,
        E_f = 10.0, ν_f = 0.3,
        r0 = 0.25, yc1 = 0.5, yc2 = 0.5
    )

    # Número de modos para a camada periódica
    N_modos_fourier = 4

    # Modos fundamentais para cada caso de "carga" 
    # da homogeneização
    ε_1 = [1.0 0.0; 
           0.0 0.0]
    ε_2 = [0.0 0.0; 
           0.0 1.0]
    ε_3 = [0.0 0.5; 
           0.5 0.0]
    modos = [ε_1, ε_2, ε_3]

    # Amostragem QMC
    N_colocacao = 1000 
    pontos = Gera_Pontos_Sobol(N_colocacao, Float64)

    # Avalia módulo de Elasticidade nos pontos de Sobol para geração de gráfico
    props = Propriedades_Material.(pontos[:, 1], pontos[:, 2], Ref(mat_params))
    E_C = first.(props)

    # Salva gráfico
    plot_Y = scatter(pontos[:, 1], pontos[:, 2], marker_z = E_C, title = "Pontos de Colocação via Sobol com E",
                     xlabel = "x", ylabel = "y", label = false, color = :jet, markersize = 3)
    savefig(plot_Y, "Resultados/pontos_colocação.pdf")

    # Vetor de redes para guardar cada rede (são 3) que vamos treinar
    redes_treinadas = Rede{Float64}[]

    # Vetor de vetores para guardar os históricos (são 3)
    historicos_treinados_ADAM = Vector{Float64}[]
    historicos_energia_ADAM = Vector{Float64}[]
    historicos_avg_ADAM = Vector{Float64}[]
    historicos_treinados_LBFGS = Vector{Float64}[]
    historicos_energia_LBFGS = Vector{Float64}[]
    historicos_avg_LBFGS = Vector{Float64}[]

    #
    # Loop pelas redes
    #
    for k in 1:3

        # Avisa que vamos treinar a rede k 
        println("\n Treinando Modo ", k)

        # Inicializa a rede
        rede = Inicializa_Rede([16, 30, 30, 30, 2], [TANH_GEN, TANH_GEN, TANH_GEN, LINEAR_GEN], Float64)
        
        # Treina a rede
        hist_ADAM, hist_energia_ADAM, hist_avg_ADAM, 
        hist_LBFGS, hist_energia_LBFGS, hist_avg_LBFGS = Treina_Rede_PINN_Energia!(rede, pontos, modos[k], N_modos_fourier, mat_params; 
                                                                 η = 0.005, epochs_ADAM = 3000, epochs_LBFGS = 6000, λ_avg = 1E6)

        # Guarda a rede no vetor de redes para fazermos o pós-processamento depois 
        push!(redes_treinadas, rede)

        # Guarda os valores do treino da rede
        push!(historicos_treinados_ADAM, hist_ADAM)
        push!(historicos_energia_ADAM, hist_energia_ADAM)
        push!(historicos_avg_ADAM, hist_avg_ADAM)
        push!(historicos_treinados_LBFGS, hist_LBFGS)
        push!(historicos_energia_LBFGS, hist_energia_LBFGS)
        push!(historicos_avg_LBFGS, hist_avg_LBFGS)

    end

    # Acompanha a evolução do objetivo ao longo do tempo
    plot_obj_treino_ADAM = plot([historicos_treinados_ADAM[i] for i in 1:3], title = "Objetivo", label = ["Rede 1" "Rede 2" "Rede 3"])
    plot_obj_energia_ADAM = plot([historicos_energia_ADAM[i] for i in 1:3], title = "Energia de Deformação", label = ["Rede 1" "Rede 2" "Rede 3"])
    plot_obj_avg_ADAM = plot([historicos_avg_ADAM[i] for i in 1:3], title = "Valor Médio dos Deslocamentos", label = ["Rede 1" "Rede 2" "Rede 3"])
    plot_obj_treino_LBFGS = plot([historicos_treinados_LBFGS[i] for i in 1:3], title = "Objetivo", label = ["Rede 1" "Rede 2" "Rede 3"])
    plot_obj_energia_LBFGS = plot([historicos_energia_LBFGS[i] for i in 1:3], title = "Energia de Deformação", label = ["Rede 1" "Rede 2" "Rede 3"])
    plot_obj_avg_LBFGS = plot([historicos_avg_LBFGS[i] for i in 1:3], title = "Valor Médio dos Deslocamentos", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_treino_ADAM, "Resultados/objetivo_treino_ADAM.pdf")
    savefig(plot_obj_energia_ADAM, "Resultados/objetivo_energia_ADAM.pdf")
    savefig(plot_obj_avg_ADAM, "Resultados/objetivo_avg_ADAM.pdf")
    savefig(plot_obj_treino_LBFGS, "Resultados/objetivo_treino_LBFGS.pdf")
    savefig(plot_obj_energia_LBFGS, "Resultados/objetivo_energia_LBFGS.pdf")
    savefig(plot_obj_avg_LBFGS, "Resultados/objetivo_avg_LBFGS.pdf")

    # Agora vamos calcular o tensor homogeneizado
    println("\n Calculando Tensor Homogeneizado ")
    CH = Calcula_Tensor_Homogeneizado(redes_treinadas, modos, N_modos_fourier, mat_params, 50)
    display(CH)

    # Calcula o tensor através da regra das misturas para comparação
    CH_mistura = Calcula_Tensor_Regra_Mistura(mat_params)
    println("\n Tensor Homogeneizado pela Regra das Misturas ")
    display(CH_mistura)

    # Grava tensor homogeneizado em um arquivo 
    writedlm("Resultados/CH.txt", CH)
    writedlm("Resultados/CH_mistura.txt", CH_mistura)
    
    # Retorna a matriz homogeneizada e os históricos
    return CH, historicos_treinados_ADAM, historicos_energia_ADAM, historicos_avg_ADAM, 
    historicos_treinados_LBFGS, historicos_energia_LBFGS, historicos_avg_LBFGS

end

# Testa ....
CH, historicos_treinados_ADAM, historicos_energia_ADAM, historicos_avg_ADAM, 
    historicos_treinados_LBFGS, historicos_energia_LBFGS, historicos_avg_LBFGS = Main_Homogenizacao()