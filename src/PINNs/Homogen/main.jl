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
include("resultados.jl")

# =============================================================================
#  Validação
# =============================================================================
function Main_Homogenizacao(mat_params::NamedTuple, N_modos_fourier::Int, N_colocacao::Int, N_eval::Int, topologia::Vector{Int}, 
                            ativ::Vector, epochs_ADAM::Int, epochs_LBFGS::Int, λ_avg::Float64; treina::Bool = true)

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

    # Inicializa as redes
    for k in 1:3
        rede = Inicializa_Rede(topologia, ativ, Float64)
        push!(redes_treinadas, rede)
    end

    # Se treina = true, vamos treinar as redes
    if treina

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
            
            # Treina a rede
            hist_ADAM, hist_energia_ADAM, hist_avg_ADAM, 
            hist_LBFGS, hist_energia_LBFGS, hist_avg_LBFGS = Treina_Rede_PINN_Energia!(redes_treinadas[k], pontos, modos[k], N_modos_fourier, mat_params; 
                                                                                       η = 0.005, epochs_ADAM, epochs_LBFGS, λ_avg)

            # Guarda os valores do treino da rede
            push!(historicos_treinados_ADAM, hist_ADAM)
            push!(historicos_energia_ADAM, hist_energia_ADAM)
            push!(historicos_avg_ADAM, hist_avg_ADAM)
            push!(historicos_treinados_LBFGS, hist_LBFGS)
            push!(historicos_energia_LBFGS, hist_energia_LBFGS)
            push!(historicos_avg_LBFGS, hist_avg_LBFGS)

        end

        # Gera os gráficos do objetivo
        Resultados!(redes_treinadas, historicos_treinados_ADAM, historicos_energia_ADAM, historicos_avg_ADAM, historicos_treinados_LBFGS,
                    historicos_energia_LBFGS, historicos_avg_LBFGS)

    # Se treina = false, vamos ler as redes treinadas de arquivos diretamente para fazer o pós-processamento
    else

        # Lê as redes treinadas de arquivos para fazer o pós-processamento
        for k in 1:3
            for (l, c) in enumerate(redes_treinadas[k].camadas)
                c.W .= readdlm("Resultados/Params/rede_$(k)_camada_$(l)_W.txt", Float64)
                c.b .= vec(readdlm("Resultados/Params/rede_$(k)_camada_$(l)_b.txt", Float64))
            end
        end

    end

    # Agora vamos calcular o tensor homogeneizado
    println("\n Calculando Tensor Homogeneizado ")
    CH = Calcula_Tensor_Homogeneizado(redes_treinadas, modos, N_modos_fourier, mat_params, N_eval)
    display(CH)

    # Calcula o tensor através da regra das misturas para comparação
    CH_mistura = Calcula_Tensor_Regra_Mistura(mat_params)
    println("\n Tensor Homogeneizado pela Regra das Misturas ")
    display(CH_mistura)

    # Grava tensor homogeneizado em um arquivo 
    writedlm("Resultados/CH.txt", CH)
    writedlm("Resultados/CH_mistura.txt", CH_mistura)
    
    # Retorna a matriz homogeneizada e os históricos
    return CH

end

# Define os hiperparâmetros do problema e roda a função principal
function Roda()

    # Propriedades dos materiais da célula
    # no caso, fibra e matriz
    # raio e centro da fibra
    mat_params = (
        E_m = 1.0, ν_m = 0.0,
        E_f = 10.0, ν_f = 0.0,
        r0 = 0.25, yc1 = 0.5, yc2 = 0.5
    )

    # Número de modos para a camada periódica
    N_modos_fourier = 4
    
    # Número de pontos de colocação via Sobol
    N_colocacao = 1000

    # Número de pontos para avaliação do tensor homogeneizado no pós-processamento
    N_eval = 500

    # Topologia da rede
    topologia = [16, 30, 30, 30, 2]

    # Ativações para cada camada
    ativ = [TANH_GEN, TANH_GEN, TANH_GEN, LINEAR_GEN]

    # Número de épocas dos otimizadores
    epochs_ADAM = 3000
    epochs_LBFGS = 1000

    # Hiperparâmetro de regularização para o valor médio dos deslocamentos
    λ_avg = 1E6

    # Testa
    CH = Main_Homogenizacao(mat_params, N_modos_fourier, N_colocacao, N_eval, topologia, ativ, epochs_ADAM, epochs_LBFGS, λ_avg; treina = false)

end

