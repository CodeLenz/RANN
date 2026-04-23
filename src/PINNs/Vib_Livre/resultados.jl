# Estima os resultados da equação diferencial a partir dos pontos de teste
function Resposta_Teste(rede:: Rede, x::Vector{Float64}, treino::NamedTuple, objetivo_treino::Vector{Float64}, 
                        perda::Vector{Vector{Float64}}, epoch::Int64, otimizador::String,
                        intervalo_monitor::Int64)

    # Atualiza pesos e bias com o resultado da otimização
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Aloca a resposta estimada
    u_test_pred = zeros(1, size(treino.t_teste, 2))
    u_analitico = zeros(1, size(treino.t_teste, 2))

    # Loop pelos dados de teste
    for i = 1:size(treino.t_teste, 2)

        # Extrai o ponto temporal para teste
        t = treino.t_teste[1, i]

        # Calcula o deslocamento pela rede
        u_test_pred[:, i] .= RNA(rede,  pesos, bias, [t])

        # Deslocamento analítico nos pontos de teste
        u_analitico = Deslocamento(treino.t_teste, treino.δ, treino.ω, treino.A, treino.ϕ)

    end

    # Grava deslocamento calculado em um arquivo para monitoramento 
    writedlm("Resultados/teste_rede_$(epoch)_$otimizador.txt", u_test_pred)

    # Acompanha a evolução do objetivo ao longo do tempo
    plot_obj_treino = plot([objetivo_treino[(epoch-(intervalo_monitor-1)):epoch]], title = "Objetivo", label = ["Treino"])
    plot_perda_inicial_u = plot([perda[1][(epoch-(intervalo_monitor-1)):epoch]], title = "Perda Inicial Deslocamento", label = ["Treino"])
    plot_perda_inicial_du = plot([perda[2][(epoch-(intervalo_monitor-1)):epoch]], title = "Perda Inicial Velocidade", label = ["Treino"])
    plot_perda_fisica = plot([perda[3][(epoch-(intervalo_monitor-1)):epoch]], title = "Perda Física", label = ["Treino"])

    plot_obj = plot(plot_obj_treino, plot_perda_inicial_u, 
                    plot_perda_inicial_du, plot_perda_fisica,
                    layout = (2, 2), size = (1000, 1000)) 

    # Grava o gráfico
    savefig(plot_obj, "Resultados/plot_obj_treino_$(epoch)_$otimizador.pdf")

    # Calcula erro entre analítico e rede neural em MAE
    erro_u = abs.((u_test_pred .- u_analitico)) 
    
    plot_erro = plot([erro_u'], title = "Erro entre Rede Neural e Analítico (Diferença)", label = ["Erro"], size = (1000, 1000))

    # Compara a resposta analítica com a calculada pela rede neural
    plot_u_teste = plot([u_analitico', u_test_pred'], title = "Deslocamento", label = ["Analítico" "Rede neural"])

    # Grava o gráfico
    savefig(plot_u_teste, "Resultados/plot_u_teste_$(epoch)_$otimizador.pdf")
    savefig(plot_erro, "Resultados/plot_erro_$(epoch)_$otimizador.pdf")

    # Retorna o deslocamento estimado
    return u_test_pred

end