# Seção transversal retangular

# Define a geometria da seção transversal
function Geometria_Retangular()

    # Lados
    H = 1.0 # [m]
    B = 1.0 # [m]

    # Coordenadas de offset da origem
    # a = b = 0 => Sem offset
    a = b = 0.0

    # Momento polar de inércia
    Je = B * H * (H^2 + B^2) / 12 # [m^4]

    # Retorna os valores
    return H, B, a, b,Je

end

# Define os pontos de colocação do domínio e também os pontos de teste
function ColocDominio_Retangular()

    # Importa os dados da seção
    H, B, a, b, _ = Geometria_Retangular()

    # Número de divisões em x e y
    div_x = 20
    div_y = 20

    # Divisões para o teste
    div_x_teste = div_x * 3
    div_y_teste = div_y * 3

    # Define um range de valores de x 
    range_x = collect(range(-B/2, B/2; length = (div_x + 2))[2:(end - 1)])
    range_x_teste = collect(range(-B/2, B/2; length = (div_x_teste + 2))[2:(end - 1)])

    # Define um range de valores de y
    range_y = collect(range(-H/2, H/2; length = (div_y+ 2))[2:(end - 1)])
    range_y_teste = collect(range(-H/2, H/2; length = (div_y_teste + 2))[2:(end - 1)])

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
            XY_fisica[1, k] = a + x

            # Coordenada y
            XY_fisica[2, k] = b + y

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
            XY_teste[1, k] = a + x

            # Coordenada y
            XY_teste[2, k] = b + y

            # Atualiza o contador
            k = k + 1

        end

    end

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
function CContorno_Retangular()

    # Importa os dados da seção
    H, B, a, b, _ = Geometria_Retangular()

    # Número de pontos de contorno por lado
    # TODO avaliar pontos repetidos nos cantos
    n_contorno_lado = 50

    # Define matriz para os pontos de contorno
    XY_contorno = zeros(Float64, 3, n_contorno_lado * 4)

    # Define contador
    k = 1

    # Define range de valores de x e y
    range_x_contorno = collect(range(-B/2, B/2; length = n_contorno_lado))
    range_y_contorno = collect(range(-H/2, H/2; length = n_contorno_lado))

    # Gera os pontos de contorno
    # Lado 1: y = - H/2
    for i in 1:n_contorno_lado

        # Coordenada x
        XY_contorno[1, i + (k-1) * n_contorno_lado] = a + range_x_contorno[i]

        # Coordenada y
        XY_contorno[2, i + (k-1) * n_contorno_lado] = b - (H/2)

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + (k-1) * n_contorno_lado] = 0.0

    end
    
    # Atualiza contador
    k = k + 1

    # Lado 2: x = B / 2
    for i in 1:n_contorno_lado

        # Coordenada x
        XY_contorno[1, i + (k-1) * n_contorno_lado] = a + B/2

        # Coordenada y
        XY_contorno[2, i + (k-1) * n_contorno_lado] = b + range_y_contorno[i]

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + (k-1) * n_contorno_lado] = 0.0

    end

    # Atualiza contador
    k = k + 1

    # Lado 3: y = H/2
    for i in 1:n_contorno_lado

        # Coordenada x
        XY_contorno[1, i + (k-1) * n_contorno_lado] = a + range_x_contorno[i]

        # Coordenada y
        XY_contorno[2, i + (k-1) * n_contorno_lado] = b + (H/2)

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + (k-1) * n_contorno_lado] = 0.0

    end

    # Atualiza contador
    k = k + 1

    # Lado 4: x = - B / 2
    for i in 1:n_contorno_lado

        # Coordenada x
        XY_contorno[1, i + (k-1) * n_contorno_lado] = a - B/2

        # Coordenada y
        XY_contorno[2, i + (k-1) * n_contorno_lado] = b + range_y_contorno[i]

        # Valor da condição de contorno no pontos
        XY_contorno[3, i + (k-1) * n_contorno_lado] = 0.0

    end

    # Salva um arquivo com o gráfico dos pontos de colocação e de teste
    plot_contorno = scatter(XY_contorno[1,:], XY_contorno[2, :], title = "Pontos de Contorno", 
                            markershape=:circle, markercolor=:blue)

    # Grava o gráfico
    savefig(plot_contorno, "Resultados/pontos_contorno.png")

    # Retorna os valores
    XY_contorno

end

# Função para cálculo da distância de contorno para aplicação forte das condições de contorno
# Proposta retirada de 
# https://medium.com/@tkadeethum/hard-constraints-in-physics-informed-neural-networks-architecture-level-enforcement-of-boundary-528e6a18bab6
# e 
# https://github.com/teeratornk/pinn_hard_constraint?tab=readme-ov-file
#=
function Distancia_Contorno_Retangular(XY::AbstractVector{T}) where T

    # Para facilitar 
    x = XY[1]
    y = XY[2]

    # Importa os dados da seção
    H, B, a, b, _ = Geometria_Retangular()

    # Calcula funções distância para cada uma das arestas
    dist_esq = x - (-B/2)    # positivo se x > x0 (inside left wall)
    dist_dir = (B/2) - x    # positive when x < x1 (inside right wall)
    dist_baixo = y - (-H/2)    # positive when y > y0 (inside bottom wall)
    dist_cima = (H/2) - y    # positive when y < y1 (inside top wall)

    # Máxima distância possível para o centro da seção
    φ_max = max( (B/2)^4, (H/2)^4 )

    # Calcula a função distância normalizada
    dist = dist_esq * dist_dir * dist_baixo * dist_cima

    # Computa e retorna a distância em relação ao contorno
    return dist

end
=#