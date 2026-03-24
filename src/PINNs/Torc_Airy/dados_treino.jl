# Dados de Treino
# Contém os dados para treino da rede neural: entradas e saídas
function Treino(prob::String)

    # Importa os pontos de colocação do domínio e os pontos de teste
    s_coloc = Symbol("ColocDominio_"*prob)
    XY_fisica, XY_teste = getfield(Main, s_coloc)()

    # Importa os pontos de contorno
    s_contorno = Symbol("CContorno_"*prob)
    XY_contorno = getfield(Main, s_contorno)()

    # Importa as condições iniciais
    t_inicial, u_inicial, du_inicial = CIniciais()
        
    # Define named tuple para guardar todos os dados do problema
    treino = (fisica = XY_fisica,
              teste = XY_teste,
              contorno  = XY_contorno,
              t_inicial = t_inicial,
              u_inicial = u_inicial,
              du_inicial = du_inicial)

    # Retorna os dados
    return treino

end

