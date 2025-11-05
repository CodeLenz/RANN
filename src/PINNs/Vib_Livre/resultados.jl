# Estima os resultados da equação diferencial a partir dos pontos de teste
function Deslocamento_Teste(rede:: Rede, x::Vector{Float64}, u_an::Matrix{Float64}, t_teste::Matrix{Float64},
                            objetivo_treino::Vector{Float64}, perda_inicial_u::Vector{Float64},
                            perda_inicial_du::Vector{Float64}, perda_fisica::Vector{Float64},
                            epoch::Int64, otimizador::String)

    # Atualiza pesos e bias com o resultado da otimização
    pesos, bias = Atualiza_pesos_bias(rede, x)

    # Aloca a resposta estimada
    u_test_pred = zeros(1, size(u_an, 2))

    # Pré-aloca a memória para sinais, que será utilizada várias vezes nesta rotina
    # a cada chamada de RNA
    #sinais = [zeros(Float64,tt) for tt in topologia] 

    # Loop pelos dados de teste
    for i = 1:size(u_an, 2)

        # Extrai o ponto temporal para teste
        t = t_teste[1, i]

        # Calcula o deslocamento pela rede
        u_test_pred[:, i] .= RNA(rede,  pesos, bias, [t])

    end

    # Grava deslocamento calculado em um arquivo para monitoramento 
    writedlm("Resultados/teste_rede_$(epoch)_$otimizador.txt", u_test_pred)

    # Acompanha a evolução do objetivo ao longo do tempo
    plot_obj_treino = plot([objetivo_treino[(epoch-999):epoch]], title = "Objetivo", label = ["Treino"])
    plot_perda_inicial_u = plot([perda_inicial_u[(epoch-999):epoch]], title = "Perda Inicial Deslocamento", label = ["Treino"])
    plot_perda_inicial_du = plot([perda_inicial_du[(epoch-999):epoch]], title = "Perda Inicial Velocidade", label = ["Treino"])
    plot_perda_fisica = plot([perda_fisica[(epoch-999):epoch]], title = "Perda Física", label = ["Treino"])

    plot_obj = plot(plot_obj_treino, plot_perda_inicial_u, 
                    plot_perda_inicial_du, plot_perda_fisica,
                    layout = (2, 2), size = (1000, 1000)) 

    # Grava o gráfico
    savefig(plot_obj, "Resultados/plot_obj_treino_$(epoch)_$otimizador.png")

    # Compara a resposta analítica com a calculada pela rede neural
    plot_u_teste = plot([u_an', u_test_pred'], title = "Deslocamento", label = ["Analítico" "Rede neural"])

    # Grava o gráfico
    savefig(plot_u_teste, "Resultados/plot_u_teste_$(epoch)_$otimizador.png")

    # Retorna o deslocamento estimado
    return u_test_pred

end