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
#  Função principal
# =============================================================================
function Main_Homogenizacao(mat_params::NamedTuple, prob::String, modos::Vector{Matrix{T}}, N_modos_fourier::Int, 
                            N_colocacao::Int, N_eval::Int, topologia::Vector{Int}, ativ::Vector, rounds::Int, epochs_ADAM::Int,  
                            epochs_LBFGS::Int, λ_avg::T; treina::Bool = true) where {T<:AbstractFloat}

    # Amostragem QMC
    pontos = Gera_Pontos_Sobol(N_colocacao, Float64)

    # Avalia módulo de Elasticidade nos pontos de Sobol para geração de gráfico
    props_simbolo = Symbol("Propriedades_Material_"*prob)
    props = getfield(Main, props_simbolo).(pontos[:, 1], pontos[:, 2], Ref(mat_params))
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
        hists_ADAM = Vector{T}[]
        hists_energ_ADAM = Vector{T}[]
        hists_avg_ADAM = Vector{T}[]
        hists_LBFGS = Vector{T}[]
        hists_energ_LBFGS = Vector{T}[]
        hists_avg_LBFGS = Vector{T}[]

        #
        # Loop pelas redes
        #
        for k in 1:3

            #=# Após o treino de uma rede, atualiza inicialização dos parâmetros da próxima rede
            if k > 1

                # Loop pelas camadas
                for i in 1:length(redes_treinadas[k].camadas)
                    redes_treinadas[k+1].camadas[i].W .= redes_treinadas[k].camadas[i].W
                    redes_treinadas[k+1].camadas[i].b .= redes_treinadas[k].camadas[i].b
                end

            end
            =#

            # Avisa que vamos treinar a rede k 
            println("\n Treinando Modo ", k)
            
            # Treina a rede
            hist_ADAM_k, hist_energ_ADAM_k, hist_avg_ADAM_k, 
            hist_LBFGS_k, hist_energ_LBFGS_k, hist_avg_LBFGS_k = Treina_Rede_PINN_Energia!(redes_treinadas[k], pontos, modos[k], N_modos_fourier, prob, mat_params; 
                                                                                       η = 0.005, rounds, epochs_ADAM, epochs_LBFGS, λ_avg)

            # Guarda os valores do treino da rede
            push!(hists_ADAM, hist_ADAM_k)
            push!(hists_energ_ADAM, hist_energ_ADAM_k)
            push!(hists_avg_ADAM, hist_avg_ADAM_k)
            push!(hists_LBFGS, hist_LBFGS_k)
            push!(hists_energ_LBFGS, hist_energ_LBFGS_k)
            push!(hists_avg_LBFGS, hist_avg_LBFGS_k)

        end

        # Gera os gráficos do objetivo
        Resultados!(redes_treinadas, hists_ADAM, hists_energ_ADAM, hists_avg_ADAM, hists_LBFGS, hists_energ_LBFGS, hists_avg_LBFGS)

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
    CH = Calcula_Tensor_Homogeneizado(redes_treinadas, modos, N_modos_fourier, prob, mat_params, N_eval)

    # Grava tensor homogeneizado em um arquivo 
    writedlm("Resultados/CH.txt", CH)

    # Compara resultados com regra das misturas para o caso retangular
    if prob == "Retangular"
        Compara_Retangular(mat_params, CH)
    end
    
    # Retorna a matriz homogeneizada
    return CH

end

# Define os hiperparâmetros do problema e roda a função principal
function Roda()

    # Propriedades dos materiais da célula - fibra e matriz
    mat_params = Dict(
        # Problema circular, raio e centro da fibra
        "Circular" => (
        E_m = 1.0, ν_m = 0.0,
        E_f = 10.0, ν_f = 0.0,
        r0 = 0.25, yc1 = 0.5, yc2 = 0.5
    ),
        # Problema retangular, altura da fibra (simétrica ao centro da célula)
        "Retangular" => (
        E_m = 1.0, ν_m = 0.3,
        E_f = 10.0, ν_f = 0.3,
        hf = 0.2
    )
    )

    # Define problema de cálculo
    prob = "Retangular"

    # Modos fundamentais para cada caso de "carga" 
    # da homogeneização
    ε_1 = [1.0 0.0; 
           0.0 0.0]
    ε_2 = [0.0 0.0; 
           0.0 1.0]
    ε_3 = [0.0 0.5; 
           0.5 0.0]
    modos = [ε_1, ε_2, ε_3]

    # Número de modos para a camada periódica
    N_modos_fourier = 4
    
    # Número de pontos de colocação via Sobol
    N_colocacao = 4000

    # Número de pontos para avaliação do tensor homogeneizado no pós-processamento
    N_eval = 500

    # Topologia da rede
    topologia = [16, 32, 2]

    # Ativações para cada camada
    ativ = [TANH_GEN, LINEAR_GEN]

    # Número de épocas dos otimizadores
    rounds = 25
    epochs_ADAM = 30
    epochs_LBFGS = 50

    # Hiperparâmetro de regularização para o valor médio dos deslocamentos
    λ_avg = 1E4

    # Testa
    CH = Main_Homogenizacao(mat_params[prob], prob, modos, N_modos_fourier, N_colocacao, N_eval, topologia, ativ, rounds, 
                            epochs_ADAM, epochs_LBFGS, λ_avg; treina = true)

end

Roda()