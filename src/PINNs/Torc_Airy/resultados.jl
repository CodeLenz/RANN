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
        u_test_pred[:, i] .= RNA(rede, pesos, bias, x)

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

    #=plot_obj = plot(plot_obj_treino, plot_perda_inicial_u, 
                    plot_perda_inicial_du, plot_perda_fisica,
                    layout = (2, 2), size = (1000, 1000)) =#

    # Layout do gráfico
    layout = @layout [a; b c]               

    # Gera o gráfico do objetivo
    plot_obj = plot(plot_obj_treino, plot_contorno, plot_perda_fisica,
                    layout = layout, size = (1000, 1000))

    # Grava o gráfico
    savefig(plot_obj, "Resultados/plot_obj_treino_$(epoch)_$otimizador.png")

    # Compara a resposta analítica com a calculada pela rede neural
    min_c = min(minimum(u_test_pred'), minimum(u_analitico))
    max_c = max(maximum(u_test_pred'), maximum(u_analitico))

    plot_u_teste_pred = scatter(treino.teste[1, :], treino.teste[2, :], marker_z = u_test_pred', clims = (min_c, max_c),
                                color = :jet, label = "Analítico", xlabel = "x", ylabel = "y")
                                
    plot_u_teste_analitico = scatter(treino.teste[1, :], treino.teste[2, :], marker_z = u_analitico', clims = (min_c, max_c),
                                     color = :jet, label = "Rede neural", xlabel = "x", ylabel = "y")
    
    plot_u_teste = plot(plot_u_teste_pred, plot_u_teste_analitico, plot_title = "Função de Airy Φ",
                        size = (2000, 1000))

    # Grava o gráfico
    savefig(plot_u_teste, "Resultados/plot_u_teste_$(epoch)_$otimizador.png")

    # Retorna o deslocamento estimado
    return u_test_pred

end