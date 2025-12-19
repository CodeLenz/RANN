# Estima os resultados da equação diferencial a partir dos pontos de teste
function Resposta_Teste(rede:: Rede, x::Vector{Float64}, treino::NamedTuple, objetivo_treino::Vector{Float64}, 
                        perda::Vector{Vector{Float64}}, epoch::Int64, prob::String, otimizador::String)

    # Atualiza pesos e bias com o resultado da otimização
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Aloca a resposta estimada e a resposta analítica
    u_test_pred = zeros(1, size(treino.teste, 2))
    u_analitico = zeros(1, size(treino.teste, 2))

    # Loop pelos dados de teste
    for i = 1:size(treino.teste, 2)

        # Extrai o ponto para teste
        x = treino.teste[:, i]

        # Calcula o deslocamento pela rede
        u_test_pred[:, i] .= RNA_forte(rede, pesos, bias, x)

        # Calcula o deslocamento Analítico
        u_analitico[:, i] .= Φ_Analitico(prob, x)

    end
    
    # Grava deslocamento calculado em um arquivo para monitoramento 
    writedlm("Resultados/teste_rede_$(epoch)_$otimizador.txt", u_test_pred)

    # Acompanha a evolução do objetivo ao longo do tempo
    plot_obj_treino = plot([objetivo_treino[(epoch-999):epoch]], title = "Objetivo", label = ["Treino"])
    plot_contorno = plot([perda[1][(epoch-999):epoch]], title = "Perda Contorno", label = ["Treino"])
    plot_perda_inicial_u = plot([perda[2][(epoch-999):epoch]], title = "Perda Inicial U", label = ["Treino"])
    plot_perda_inicial_du = plot([perda[3][(epoch-999):epoch]], title = "Perda Inicial dU", label = ["Treino"])
    plot_perda_fisica = plot([perda[4][(epoch-999):epoch]], title = "Perda Física", label = ["Treino"])

    # Gráfico da função objetivo
    # Layout do gráfico
    layout = @layout [a; b c]               

    # Gera o gráfico
    plot_obj = plot(plot_obj_treino, plot_contorno, plot_perda_fisica,
                    layout = layout, size = (1000, 1000))

    # Grava o gráfico
    savefig(plot_obj, "Resultados/plot_obj_treino_$(epoch)_$otimizador.png")

    # Compara a resposta analítica com a calculada pela rede neural
    # Define escala única dos gráficos
    min_c = min(minimum(u_test_pred), minimum(u_analitico))
    max_c = max(maximum(u_test_pred), maximum(u_analitico))

    # Calcula erro entre analítico e rede neural em MAE
    erro_u = abs.((u_test_pred .- u_analitico))

    # Rede neural
    plot_u_teste_pred = scatter(treino.teste[1, :], treino.teste[2, :], marker_z = u_test_pred', 
                                clims = (min_c, max_c), title = "Rede Neural", xlabel = "x", ylabel = "y",
                                label = false, color = :jet, markersize = 8, markerstrokecolor = :black, 
                                markerstrokewidth = 0.2, alpha = 0.9)
    
    # Analítico
    plot_u_teste_analitico = scatter(treino.teste[1, :], treino.teste[2, :], marker_z = u_analitico', 
                                     clims = (min_c, max_c), title = "Analítico", xlabel = "x", ylabel = "y",
                                     label = false, color = :jet, markersize = 8, markerstrokecolor = :black, 
                                     markerstrokewidth = 0.2, alpha = 0.9)

    # Erro
    plot_erro = scatter(treino.teste[1, :], treino.teste[2, :], marker_z = erro_u', 
                        title = "Erro entre Rede Neural e Analítico (Diferença)", xlabel = "x", ylabel = "y",
                        label = false, clims = (0.0, maximum(erro_u')), markersize = 8, markerstrokecolor = :black, 
                        markerstrokewidth = 0.2, alpha = 0.9, size = (1000, 1000), c = cgrad(:jet), colorbar = true)
    
    plot_u_teste = plot(plot_u_teste_pred, plot_u_teste_analitico, plot_title = "Função de Airy Φ",
                        size = (3000, 1000))

    # Grava o gráfico
    savefig(plot_u_teste, "Resultados/plot_u_teste_$(epoch)_$otimizador.png")
    savefig(plot_erro, "Resultados/plot_erro_$(epoch)_$otimizador.png")

    # Retorna o deslocamento estimado
    return u_test_pred

end