# Função para geração de gráficos da função objetivo
function Resultados!(redes::Vector{Rede{T}}, historicos_treinados_ADAM::Vector{Vector{T}}, historicos_energia_ADAM::Vector{Vector{T}}, 
                     historicos_avg_ADAM::Vector{Vector{T}}, historicos_treinados_LBFGS::Vector{Vector{T}},
                     historicos_energia_LBFGS::Vector{Vector{T}}, historicos_avg_LBFGS::Vector{Vector{T}}) where {T<:AbstractFloat}

    # Grava os parâmetros de treino em arquivos para uso posterior
    for k in 1:3
        for (l, c) in enumerate(redes[k].camadas)
            writedlm("Resultados/Params/rede_$(k)_camada_$(l)_W.txt", c.W)
            writedlm("Resultados/Params/rede_$(k)_camada_$(l)_b.txt", c.b)
        end
    end
    
    # ADAM
    # Objetivo
    plot_obj_treino_ADAM = plot([historicos_treinados_ADAM[i] for i in 1:3], title = "Objetivo", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_treino_ADAM, "Resultados/objetivo_treino_ADAM.pdf")

    # Perda de energia
    plot_obj_energia_ADAM = plot([historicos_energia_ADAM[i] for i in 1:3], title = "Energia de Deformação", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_energia_ADAM, "Resultados/objetivo_energia_ADAM.pdf")

    # Valor médio dos deslocamentos
    plot_obj_avg_ADAM = plot([historicos_avg_ADAM[i] for i in 1:3], title = "Valor Médio dos Deslocamentos", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_avg_ADAM, "Resultados/objetivo_avg_ADAM.pdf")

    # L-BFGS
    # Objetivo
    plot_obj_treino_LBFGS = plot([historicos_treinados_LBFGS[i] for i in 1:3], title = "Objetivo", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_treino_LBFGS, "Resultados/objetivo_treino_LBFGS.pdf")

    # Perda de energia
    plot_obj_energia_LBFGS = plot([historicos_energia_LBFGS[i] for i in 1:3], title = "Energia de Deformação", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_energia_LBFGS, "Resultados/objetivo_energia_LBFGS.pdf")

    # Valor médio dos deslocamentos
    plot_obj_avg_LBFGS = plot([historicos_avg_LBFGS[i] for i in 1:3], title = "Valor Médio dos Deslocamentos", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_avg_LBFGS, "Resultados/objetivo_avg_LBFGS.pdf")

end