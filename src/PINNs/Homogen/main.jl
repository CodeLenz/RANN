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

include("ativ.jl")
include("RNA.jl")
include("PINN.jl")
include("material.jl")
include("tensao_pos.jl")
include("treino.jl")
include("AdamW.jl")

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
    historicos_treinados = Vector{Float64}[]
    historicos_energia = Vector{Float64}[]
    historicos_avg = Vector{Float64}[]

    #
    # Loop pelas redes
    #
    for k in 1:3

        # Avisa que vamos treinar a rede k 
        println("\n Treinando Modo ", k)

        # Inicializa a rede
        rede = Inicializa_Rede([16, 40, 40, 2], [TANH_GEN, TANH_GEN, LINEAR_GEN], Float64)
        
        # Treina a rede
        hist, hist_energia, hist_avg = Treina_Rede_PINN_Energia!(rede, pontos, modos[k], N_modos_fourier, mat_params; η = 0.005, epochs = 5000, λ_avg = 1E6)

        # Guarda a rede no vetor de redes para fazermos o pós-processamento depois 
        push!(redes_treinadas, rede)

        # Guarda os valores do treino da rede
        push!(historicos_treinados, hist)
        push!(historicos_energia, hist_energia)
        push!(historicos_avg, hist_avg)

    end

    # Acompanha a evolução do objetivo ao longo do tempo
    plot_obj_treino = plot([historicos_treinados[i] for i in 1:3], title = "Objetivo", label = ["Rede 1" "Rede 2" "Rede 3"])
    plot_obj_energia = plot([historicos_energia[i] for i in 1:3], title = "Energia de Deformação", label = ["Rede 1" "Rede 2" "Rede 3"])
    plot_obj_avg = plot([historicos_avg[i] for i in 1:3], title = "Valor Médio dos Deslocamentos", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_treino, "Resultados/objetivo_treino.pdf")
    savefig(plot_obj_energia, "Resultados/objetivo_energia.pdf")
    savefig(plot_obj_avg, "Resultados/objetivo_avg.pdf")

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
    return CH, historicos_treinados

end

# Testa ....
CH, historicos_treinados = Main_Homogenizacao()