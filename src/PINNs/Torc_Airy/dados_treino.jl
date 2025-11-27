# Dados de Treino
# Contém os dados para treino da rede neural: entradas e saídas
function Treino(prob::String)

    # Importa os pontos de colocação do domínio e os pontos de teste
    XY_fisica, XY_teste = ColocDominio(prob)

    # Importa os pontos de contorno
    XY_contorno = CContorno(prob)

    # Importa as condições iniciais
    t_inicial, u_inicial, du_inicial = CIniciais(prob)
        
    # Define dicionário para guardar todos os dados do problema
    treino = (fisica = XY_fisica,
                   teste = XY_teste,
                   contorno  = XY_contorno,
                   t_inicial = t_inicial,
                   u_inicial = u_inicial,
                   du_inicial = du_inicial)

    # Retorna os dados
    return treino

end

