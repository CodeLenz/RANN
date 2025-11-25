# Define as condições de contorno iniciais
function CIniciais(prob::String)

    # Caso 1: Seção transversal circular
    if prob == "circular"

        # Nesse caso, não temos condições iniciais, portanto deve retornar matrizes vazias
        t_inicial = nothing
        u_inicial = nothing
        du_inicial = nothing

    # Caso não seja selecionado nenhum problema, define matriz vazia
    else

        t_inicial = nothing
        u_inicial = nothing
        du_inicial = nothing

    end

    # Retorna os valores
    return t_inicial, u_inicial, du_inicial

end