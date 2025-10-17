# Estima os resultados da equação diferencial a partir dos pontos de teste
function Deslocamento_Teste(rede:: Rede, x::Vector{Float64}, u_an::Matrix{Float64}, t_teste::Matrix{Float64},
                            objetivo_treino::Vector{Float64}, epoch::Int64)

    # Alias
    # topologia = rede.topologia

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
    writedlm("Resultados/teste_rede_$(epoch).txt", u_test_pred)

    # Acompanha a evolução do objetivo ao longo do tempo
    plot_obj_treino = plot([objetivo_treino[1:epoch]], title = "Objetivo", label = ["Treino"])

    # Grava o gráfico
    savefig(plot_obj_treino, "Resultados/plot_obj_treino_$(epoch).png")

    # Compara a resposta analítica com a calculada pela rede neural
    plot_u_teste = plot([u_an', u_test_pred'], title = "Deslocamento", label = ["Analítico" "Rede neural"])

    # Grava o gráfico
    savefig(plot_u_teste, "Resultados/plot_u_teste_$(epoch).png")

    # Retorna o deslocamento estimado
    return u_test_pred

end