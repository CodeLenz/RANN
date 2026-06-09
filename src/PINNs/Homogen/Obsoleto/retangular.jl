# Célula unitária
# Dimensões [0; 1] x [0; 1], definida em torno da origem (0, 0)

# Define a geometria da Célula unitária
function Geometria_Celula()

    # Lados
    y1 = 1.0
    y2 = 1.0

    # Retorna os valores
    return y1, y2

end

# Define os pontos de colocação do domínio e também os pontos de teste para cálculo da tensão
function ColocDominio_Retangular(h = 0.01)

    # Importa os dados da célula unitária
    y1, y2 = Geometria_Celula()

    # Número de divisões em y1 e y2
    div_y1 = 20
    div_y2 = 20

    # Número de pontos para cálculo da tensão
    div_y1_teste = div_y1 * 5
    div_y2_teste = div_y2 * 5

    # Define um range de valores de y1 e y2 
    # Limites consideram a vizinhança h dos pontos para cálculo dos deslocamentos flutuantes
    range_y1 = collect(range(h, 1 - h; length = div_y1))
    range_y2 = collect(range(h, 1 - h; length = div_y2))

    # Define um range de valores para os pontos de tensão
    range_y1_teste = collect(range(h, 1 - h; length = div_y1_teste))
    range_y2_teste = collect(range(h, 1 - h; length = div_y2_teste))

    # Define matriz para os pontos de perda física
    XY_fisica = zeros(Float64, 2, div_y1 * div_y2)

    # Define matriz para os pontos de teste
    XY_teste = zeros(Float64, 2, div_y1_teste * div_y2_teste)

    # Gera os pontos de colocação
    # Contador
    k = 1

    # Loop em x
    for y1 in range_y1

        # Loop em y
        for y2 in range_y2

            # Coordenada x
            XY_fisica[1, k] = y1

            # Coordenada y
            XY_fisica[2, k] = y2

            # Atualiza o contador
            k = k + 1
            
        end

    end

    # Gera os pontos de teste
    # Reseta o contador
    k = 1

    # Loop em x
    for y1 in range_y1_teste

        # Loop em y
        for y2 in range_y2_teste

            # Coordenada x
            XY_teste[1, k] = y1

            # Coordenada y
            XY_teste[2, k] = y2

            # Atualiza o contador
            k = k + 1

        end

    end

    # Salva um arquivo com o gráfico dos pontos de colocação e de teste
    plot_XY = scatter(XY_fisica[1,:], XY_fisica[2, :], title = "Pontos de Colocação e Tensão", 
                   label = "Colocação", markershape=:circle, markercolor=:blue)
    scatter!(plot_XY, XY_teste[1,:], XY_teste[2, :], label="Teste", markershape=:cross, markercolor=:red)

    # Grava o gráfico
    savefig(plot_XY, "Resultados/pontos_colocação_teste.pdf")

    # Retorna os valores
    XY_fisica, XY_teste
    
end