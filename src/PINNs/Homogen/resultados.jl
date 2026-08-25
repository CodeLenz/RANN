# Função para geração de gráficos da função objetivo
function Resultados!(redes::Vector{Rede{T}}, historicos_treinados_ADAM::Vector{Vector{T}}, 
                     historicos_treinados_LBFGS::Vector{Vector{T}}) where {T<:AbstractFloat}

    # Grava os parâmetros de treino em arquivos para uso posterior
    for k in 1:3
        for (l, c) in enumerate(redes[k].camadas)
            writedlm("Resultados/Params/rede_$(k)_camada_$(l)_W.txt", c.W)
            writedlm("Resultados/Params/rede_$(k)_camada_$(l)_b.txt", c.b)
        end
    end
    
    # ADAM - Objetivo
    plot_obj_treino_ADAM = plot([historicos_treinados_ADAM[i] for i in 1:3], title = "Objetivo - ADAM", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_treino_ADAM, "Resultados/objetivo_treino_ADAM.pdf")

    # L-BFGS - Objetivo
    plot_obj_treino_LBFGS = plot([historicos_treinados_LBFGS[i] for i in 1:3], title = "Objetivo - L-BFGS", label = ["Rede 1" "Rede 2" "Rede 3"])
    savefig(plot_obj_treino_LBFGS, "Resultados/objetivo_treino_LBFGS.pdf")

end