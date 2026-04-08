# Seção transversal em L; Origem no canto inferior esquerdo

#
#        b
#        _
#       | |
#       | |
#    a  | |
#       | |________
#       |__________|  b
#
#            a
#

# Define a geometria da seção transversal
function Geometria_L()

    # Lados
    a = 10.0 # [m]
    b = 1.0 # [m]

    # Coordenadas de offset da origem
    # off_x = off_y = 0 => Sem offset
    off_x = off_y = 0.0

    # Retorna os valores
    return a, b, off_x, off_y

end

# Função auxiliar para verificar se um ponto está dentro da seção
function ponto_dentro_L(x, y, a, b)

    # Verifica se o ponto está dentro da seção
    if (x >= 0 && x <= a && y >= 0 && y <= b) || (x >= 0 && x <= b && y >= 0 && y <= a)
        return true
    else
        return false
    end

end

# Define os pontos de colocação do domínio e também os pontos de teste
function ColocDominio_L()

    # Importa os dados da seção
    a, b, off_x, off_y = Geometria_L()

    # Fase 1:
    # Define pontos de colocação assumindo uma distribuição uniforme em um quadrado de lado a
    
    # Número de divisões em x e y
    div_x = 35
    div_y = 35

    # Divisões para o teste
    div_x_teste = div_x * 3
    div_y_teste = div_y * 3

    # Define um range de valores de x 
    range_x = collect(range(0, a; length = (div_x + 2))[2:(end - 1)])
    range_x_teste = collect(range(0, a; length = (div_x_teste + 2))[2:(end - 1)])

    # Define um range de valores de y
    range_y = collect(range(0, a; length = (div_y+ 2))[2:(end - 1)])
    range_y_teste = collect(range(0, a; length = (div_y_teste + 2))[2:(end - 1)])

    # Define matriz para os pontos de perda física
    XY_fisica = zeros(Float64, 2, div_x * div_y)

    # Define matriz para os pontos de teste
    XY_teste = zeros(Float64, 2, div_x_teste * div_y_teste)

    # Gera os pontos de colocação
    # Contador
    k = 1

    # Loop em x
    for x in range_x

        # Loop em y
        for y in range_y

            # Coordenada x
            XY_fisica[1, k] = off_x + x

            # Coordenada y
            XY_fisica[2, k] = off_y + y

            # Atualiza o contador
            k = k + 1
            
        end

    end

    # Gera os pontos de teste
    # Reseta o contador
    k = 1

    # Loop em x
    for x in range_x_teste

        # Loop em y
        for y in range_y_teste

            # Coordenada x
            XY_teste[1, k] = off_x + x

            # Coordenada y
            XY_teste[2, k] = off_y + y

            # Atualiza o contador
            k = k + 1

        end

    end

    # Fase 2:
    # Remove os pontos que estão fora da seção em L

    # Pontos de colocação física
    # Encontra os indices onde temos pontos válidos da seção
    indices_fisica = findall(λ -> ponto_dentro_L(XY_fisica[1, λ], XY_fisica[2, λ], a, b), 1:size(XY_fisica, 2))

    # Filtra essas colunas
    XY_fisica = XY_fisica[:, indices_fisica]

    # Pontos de teste
    indices_teste = findall(λ -> ponto_dentro_L(XY_teste[1, λ], XY_teste[2, λ], a, b), 1:size(XY_teste, 2))

    # Filtra essas colunas
    XY_teste = XY_teste[:, indices_teste]

    @show size(XY_fisica), size(XY_teste)

    # Salva um arquivo com o gráfico dos pontos de colocação e de teste
    plot_XY = scatter(XY_fisica[1,:], XY_fisica[2, :], title = "Pontos de Colocação e Contorno", 
                   label = "Colocação", markershape=:circle, markercolor=:blue)
    scatter!(plot_XY, XY_teste[1,:], XY_teste[2, :], label="Teste", markershape=:cross, markercolor=:red)

    # Grava o gráfico
    savefig(plot_XY, "Resultados/pontos_colocação_teste.png")

    # Retorna os valores
    XY_fisica, XY_teste
    
end

# Define os pontos de condição de contorno
function CContorno_L()

    # Importa os dados da seção
    a, b, off_x, off_y = Geometria_L()

    # Número de pontos de contorno por lado
    # TODO avaliar pontos repetidos nos cantos
    n_contorno_lado_a = 50
    n_contorno_lado_b = 5
    n_contorno_lado_ab = n_contorno_lado_a - n_contorno_lado_b

    # Define matriz para os pontos de contorno
    XY_contorno = zeros(Float64, 3, n_contorno_lado_a * 2 + n_contorno_lado_b * 2 + n_contorno_lado_ab * 2)

    # Define contador
    k = 0

    # Define range de valores de x e y
    range_contorno_a = collect(range(0, a; length = n_contorno_lado_a))
    range_contorno_b = collect(range(0, b; length = n_contorno_lado_b))
    range_contorno_ab = collect(range(b, a; length = n_contorno_lado_a - n_contorno_lado_b))

    # Gera os pontos de contorno
    # Lado 1: y = 0
    for i in 1:n_contorno_lado_a

        # Coordenada x
        XY_contorno[1, i + k] = off_x + range_contorno_a[i]

        # Coordenada y
        XY_contorno[2, i + k] = off_y

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + k] = 0.0

    end

    # Atualiza contador
    k = k + n_contorno_lado_a

    # Lado 2: x = a
    for i in 1:n_contorno_lado_b

        # Coordenada x
        XY_contorno[1, i + k] = off_x + a

        # Coordenada y
        XY_contorno[2, i + k] = off_y + range_contorno_b[i]

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + k] = 0.0

    end

    # Atualiza contador
    k = k + n_contorno_lado_b

    # Lado 3: y = b
    for i in 1:n_contorno_lado_ab

        # Coordenada x
        XY_contorno[1, i + k] = off_x + range_contorno_ab[i]

        # Coordenada y
        XY_contorno[2, i + k] = off_y + b

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + k] = 0.0

    end

    # Atualiza contador
    k = k + n_contorno_lado_ab

    # Lado 4: x = b
    for i in 1:n_contorno_lado_ab

        # Coordenada x
        XY_contorno[1, i + k] = off_x + b

        # Coordenada y
        XY_contorno[2, i + k] = off_y + range_contorno_ab[i]

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + k] = 0.0

    end

    # Atualiza contador
    k = k + n_contorno_lado_ab

    # Lado 5: y = a
    for i in 1:n_contorno_lado_b

        # Coordenada x
        XY_contorno[1, i + k] = off_x + range_contorno_b[i]

        # Coordenada y
        XY_contorno[2, i + k] = off_y + a

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + k] = 0.0

    end

    # Atualiza contador
    k = k + n_contorno_lado_b

    # Lado 6: x = 0
    for i in 1:n_contorno_lado_a

        # Coordenada x
        XY_contorno[1, i + k] = off_x 

        # Coordenada y
        XY_contorno[2, i + k] = off_y + range_contorno_a[i]

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + k] = 0.0

    end

    # Salva um arquivo com o gráfico dos pontos de colocação e de teste
    plot_contorno = scatter(XY_contorno[1,:], XY_contorno[2, :], title = "Pontos de Contorno", 
                            markershape=:circle, markercolor=:blue)

    # Grava o gráfico
    savefig(plot_contorno, "Resultados/pontos_contorno.png")

    # Retorna os valores
    XY_contorno

end