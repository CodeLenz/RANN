# Define a geometria da seção transversal
function Geometria(prob::String)

    # Seção circular
    if prob == "circular"

        # Raio
        R = 1.0 #[m]

        # Momento polar de inércia
        Je = π * R^4 / 2 # [m^4]

        # Retorna os valores
        return R, Je

    # Seção quadrada
    elseif prob == "quad"

        # Lado
        L = 1.0 # [m]

        # Momento polar de inércia
        Je = L^4 / 6 # [m^4]

        # Retorna os valores
        return L, Je

    end

end

# Define os pontos de colocação do domínio e também os pontos de teste
function ColocDominio(prob::String)

    # Caso 1: Seção transversal circular
    if prob == "circular"

        # Importa os dados da seção
        R, _ = Geometria(prob)

        # Coordenadas de offset da origem
        # a = b = 0 => Sem offset
        a = b = 0.0

        # Número de divisões em raio e ângulo
        div_r = 10
        div_θ = 16

        # Divisões para o teste
        div_r_teste = div_r * 3
        div_θ_teste = div_θ * 3

        # Define um range de raios 
        raios = collect(range(0, R; length = div_r + 1)[2:end])
        raios_teste = collect(range(0, R; length = div_r_teste + 1)[2:end])

        # Define um range de ângulos
        angulos = collect(range(0, 2π; length = div_θ + 1)[1:end-1])
        angulos_teste = collect(range(0, 2π; length = div_θ_teste + 1)[1:end-1])

        # Define matriz para os pontos de perda física
        XY_fisica = zeros(Float64, 2, div_r * div_θ)

        # Define matriz para os pontos de teste
        XY_teste = zeros(Float64, 2, div_r_teste * div_θ_teste)

        # Gera os pontos de colocação
        # Contador
        k = 1

        # Loop pelos raios
        for r in raios

            # Loop pelos ângulos
            for θ in angulos

                # Coordenada x
                XY_fisica[1, k] = a + r * cos(θ)

                # Coordenada y
                XY_fisica[2, k] = b + r * sin(θ)

                # Atualiza o contador
                k = k + 1
                
            end

        end

        # Gera os pontos de teste
        # Reseta o contador
        k = 1

        # Loop pelos raios
        for r in raios_teste

            # Loop pelos ângulos
            for θ in angulos_teste

                # Coordenada x
                XY_teste[1, k] = a + r * cos(θ)

                # Coordenada y
                XY_teste[2, k] = b + r * sin(θ)

                # Atualiza o contador
                k = k + 1

            end

        end

    # Caso 2: Seção transversal circular
    elseif prob == "quad"

        # Importa os dados da seção
        L, _ = Geometria(prob)

        # Coordenadas de offset da origem
        # a = b = 0 => Sem offset
        a = b = 0.0

        # Número de divisões em x e y
        div_x = 10
        div_y = 10

        # Divisões para o teste
        div_x_teste = div_x * 3
        div_y_teste = div_y * 3

        # Define um range de valores de x 
        range_x = collect(range(-L/2, L/2; length = (div_x + 2))[2:(end - 1)])
        range_x_teste = collect(range(-L/2, L/2; length = (div_x_teste + 2))[2:(end - 1)])

        # Define um range de valores de y
        range_y = collect(range(-L/2, L/2; length = (div_y+ 2))[2:(end - 1)])
        range_y_teste = collect(range(-L/2, L/2; length = (div_y_teste + 2))[2:(end - 1)])

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
        

    # Caso não seja selecionado nenhum problema, define matrizes vazias
    else

        XY_fisica, XY_teste = nothing, nothing

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
function CContorno(prob::String)

    # Caso 1: Seção transversal circular
    if prob == "circular"

        # Importa os dados da seção
        R, _ = Geometria(prob)

        # Coordenadas de offset da origem
        # a = b = 0 => Sem offset 
        a = b = 0.0

        # Número de pontos de contorno
        n_contorno = 500

        # Define um range de ângulos
        θ = collect(range(0.0, 2.0 * pi, n_contorno))

        # Define matriz para os pontos de contorno
        XY_contorno = zeros(Float64, 3, n_contorno)

        # Gera os pontos de contorno
        for i in 1:n_contorno

            # Coordenada x
            XY_contorno[1, i] = a + R * cos(θ[i])

            # Coordenada y
            XY_contorno[2, i] = b + R * sin(θ[i])

            # Valor da condição de contorno no pontos
            XY_contorno[3, i] = 0.0

        end

        # Salva um arquivo com o gráfico dos pontos de colocação e de teste
        plot_contorno = scatter(XY_contorno[1,:], XY_contorno[2, :], title = "Pontos de Contorno", 
                                markershape=:circle, markercolor=:blue)

        # Grava o gráfico
        savefig(plot_contorno, "Resultados/pontos_contorno.png")

    # Caso 2: Seção quadrada
    elseif prob == "quad"

        # Importa os dados da seção
        L, _ = Geometria(prob)

        # Coordenadas de offset da origem
        # a = b = 0 => Sem offset 
        a = b = 0.0

        # Número de pontos de contorno por lado
        # TODO avaliar pontos repetidos nos cantos
        n_contorno_lado = 50

        # Define matriz para os pontos de contorno
        XY_contorno = zeros(Float64, 3, n_contorno_lado * 4)

        # Define contador
        k = 1

        # Define range de valores de x e y
        range_x_contorno = collect(range(-L/2, L/2; length = n_contorno_lado))
        range_y_contorno = collect(range(-L/2, L/2; length = n_contorno_lado))

        # Gera os pontos de contorno
        # Lado 1: y = - L/2
        for i in 1:n_contorno_lado

            # Coordenada x
            XY_contorno[1, i + (k-1) * n_contorno_lado] = a + range_x_contorno[i]

            # Coordenada y
            XY_contorno[2, i + (k-1) * n_contorno_lado] = b - (L/2)

            # Valor da condição de contorno no pontos
            XY_contorno[3, i + (k-1) * n_contorno_lado] = 0.0

        end
        
        # Atualiza contador
        k = k + 1

        # Lado 2: x = L / 2
        for i in 1:n_contorno_lado

            # Coordenada x
            XY_contorno[1, i + (k-1) * n_contorno_lado] = a + L/2

            # Coordenada y
            XY_contorno[2, i + (k-1) * n_contorno_lado] = b + range_y_contorno[i]

            # Valor da condição de contorno no pontos
            XY_contorno[3, i + (k-1) * n_contorno_lado] = 0.0

        end

        # Atualiza contador
        k = k + 1

        # Lado 3: y = L/2
        for i in 1:n_contorno_lado

            # Coordenada x
            XY_contorno[1, i + (k-1) * n_contorno_lado] = a + range_x_contorno[i]

            # Coordenada y
            XY_contorno[2, i + (k-1) * n_contorno_lado] = b + (L/2)

            # Valor da condição de contorno no pontos
            XY_contorno[3, i + (k-1) * n_contorno_lado] = 0.0

        end

        # Atualiza contador
        k = k + 1

        # Lado 4: x = - L / 2
        for i in 1:n_contorno_lado

            # Coordenada x
            XY_contorno[1, i + (k-1) * n_contorno_lado] = a - L/2

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

    # Caso não seja selecionado nenhum problema, define entrada vazia
    else

        XY_contorno = nothing

    end    

    # Retorna os valores
    XY_contorno

end

# Função para cálculo da distância de contorno para aplicação forte das condições de contorno
function Distancia_Contorno(XY::AbstractVector{T}, prob::String) where T

    # Para facilitar 
    x = XY[1]
    y = XY[2]

    # Caso 1: Seção transversal circular
    if prob == "circular"

        # Importa os dados da seção
        R, _ = Geometria(prob)

        # Coordenadas de offset da origem
        # a = b = 0 => Sem offset
        a = b = 0.0

        # calcula a função distância suave
        dist = 1.0 - ((x - a)^2 + (y - b)^2) / R^2

    # Caso 2: Seção quadrada
    elseif prob == "quad"

        # Importa os dados da seção
        L, _ = Geometria(prob)

        # Coordenadas de offset da origem
        # a = b = 0 => Sem offset
        a = b = 0.0

        # Calcula funções distância para cada uma das arestas
        dist_esq = x - (-L/2)    # positivo se x > x0 (inside left wall)
        dist_dir = (L/2) - x    # positive when x < x1 (inside right wall)
        dist_baixo = y - (-L/2)    # positive when y > y0 (inside bottom wall)
        dist_cima = (L/2) - y    # positive when y < y1 (inside top wall)

        # Máxima distância possível para o centro da seção
        φ_max = (L/2)^4

        # Calcula a função distância normalizada
        dist = dist_esq * dist_dir * dist_baixo * dist_cima

    end

    # Computa e retorna a distância em relação ao contorno
    return dist

end