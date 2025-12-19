# Define a geometria da seção transversal
function Geometria_Circular()

    # Raio
    R = 1.0 #[m]

    # Momento polar de inércia
    Je = π * R^4 / 2 # [m^4]

    # Retorna os valores
    return R, Je

end

# Define os pontos de colocação do domínio e também os pontos de teste
function ColocDominio(prob::String)

    # Caso 1: Seção transversal circular
    if prob == "circular"

        # Importa os dados da seção
        R, _ = Geometria_Circular()

        a = 2 * R
        b = 2 * R 

        # Número de divisões em raio e ângulo
        div_r = 10
        div_θ = 16

        # Divisões para o teste
        div_r_teste = div_r * 3
        div_θ_teste = div_θ * 3

        # Define um range de raios 
        raios = raios = collect(range(0, R; length = div_r + 1)[2:end])
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
        R, _ = Geometria_Circular()

        a = 2 * R
        b = 2 * R 

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

    # Caso não seja selecionado nenhum problema, define entrada vazia
    else

        XY_contorno = nothing

    end    

    # Retorna os valores
    XY_contorno

end

# Função para cálculo da distância de contorno para aplicação forte das condições de contorno
function Distancia_Contorno(XY::Vector{T}) where T

    # Importa os dados da seção
    R, _ = Geometria_Circular()

    a = 2 * R
    b = 2 * R

    # Computa e retorna a distância em relação ao contorno
    return R - sqrt((XY[1]-a)^2 + (XY[2]-b)^2)

end