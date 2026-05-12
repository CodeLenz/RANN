# Estima os resultados da equação diferencial a partir dos pontos de teste
function Resposta_Teste(rede:: Rede, x::Vector{Float64}, treino::NamedTuple, objetivo_treino::Vector{Float64}, 
                        epoch::Int64, prob::String, otimizador::String, intervalo_monitor::Int64)

    # Atualiza pesos e bias com o resultado da otimização
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Aloca a resposta estimada
    u_test_pred = zeros(1, size(treino.teste, 2))

    # Loop pelos dados de teste
    for i = 1:size(treino.teste, 2)

        # Extrai o ponto para teste
        x_i = treino.teste[:, i]

        # Calcula o deslocamento pela rede
        u_test_pred[:, i] .= RNA_forte(rede, pesos, bias, x_i, prob)

    end

    # Calcula o deslocamento Analítico
    u_analitico = Φ_Analitico(prob, treino.teste)
    
    # Grava deslocamento calculado em um arquivo para monitoramento 
    writedlm("Resultados/teste_rede_$(epoch)_$otimizador.txt", u_test_pred)

    # Acompanha a evolução do objetivo ao longo do tempo
    plot_obj_treino = plot([objetivo_treino[(epoch-(intervalo_monitor-1)):epoch]], title = "Objetivo", label = ["Treino"])

    # Gráfico da função objetivo
    # Neste momento, temos apenas a perda física
    plot_obj = plot(plot_obj_treino, size = (1000, 1000))

    # Grava o gráfico
    savefig(plot_obj, "Resultados/plot_obj_treino_$(epoch)_$otimizador.pdf")

    # Compara a resposta analítica com a calculada pela rede neural
    # Define escala única dos gráficos
    min_c = min(minimum(u_test_pred), minimum(u_analitico))
    max_c = max(maximum(u_test_pred), maximum(u_analitico))

    # Calcula erro entre analítico e rede neural em MAE
    erro_u = abs.((u_test_pred .- u_analitico))

    # Rede neural
    plot_u_teste_pred = scatter(treino.teste[1, :], treino.teste[2, :], marker_z = u_test_pred[:], 
                                clims = (min_c, max_c), title = "Rede Neural", xlabel = "x", ylabel = "y",
                                label = false, color = :jet, markersize = 1, markerstrokecolor = :black,
                                markerstrokewidth = 0.0, alpha = 0.9)
    
    # Analítico
    plot_u_teste_analitico = scatter(treino.teste[1, :], treino.teste[2, :], marker_z = u_analitico', 
                                     clims = (min_c, max_c), title = "Analítico", xlabel = "x", ylabel = "y",
                                     label = false, color = :jet, markersize = 1, markerstrokecolor = :black, 
                                     markerstrokewidth = 0.0, alpha = 0.9)

    # Erro
    plot_erro = scatter(treino.teste[1, :], treino.teste[2, :], marker_z = erro_u', 
                        title = "Erro entre Rede Neural e Analítico (Diferença)", xlabel = "x", ylabel = "y",
                        label = false, clims = (0.0, maximum(erro_u')), markersize = 1, markerstrokecolor = :black, 
                        markerstrokewidth = 0.0, alpha = 0.9, size = (1000, 1000), c = cgrad(:jet), colorbar = true)
    
    # Gráfico completo
    plot_u_teste = plot(plot_u_teste_pred, plot_u_teste_analitico, plot_title = "Função de Airy Φ",
                        size = (3000, 1000))
    
    # Grava o gráfico
    savefig(plot_u_teste, "Resultados/plot_u_teste_$(epoch)_$otimizador.pdf")
    savefig(plot_erro, "Resultados/plot_erro_$(epoch)_$otimizador.pdf")

    # Retorna o deslocamento estimado
    return u_test_pred

end