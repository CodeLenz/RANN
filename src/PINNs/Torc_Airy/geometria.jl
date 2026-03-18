# Define os pontos de colocação do domínio e também os pontos de teste
function ColocDominio(prob::String)

    # Caso 1: Seção transversal circular
    if prob == "circular"

        XY_fisica, XY_teste = ColocDominio_Circular()

    # Caso 2: Seção transversal retangular
    elseif prob == "retang"

        XY_fisica, XY_teste = ColocDominio_Retangular() 
        
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

        XY_contorno = CContorno_Circular()

    # Caso 2: Seção retangular
    elseif prob == "retang"

        XY_contorno = CContorno_Retangular()

    # Caso não seja selecionado nenhum problema, define entrada vazia
    else

        XY_contorno = nothing

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
function Distancia_Contorno(XY::AbstractVector{T}, prob::String) where T

    # Caso 1: Seção transversal circular
    if prob == "circular"

        dist = Distancia_Contorno_Circular(XY)

    # Caso 2: Seção retangular
    elseif prob == "retang"

        dist = Distancia_Contorno_Retangular(XY)

    end

    # Computa e retorna a distância em relação ao contorno
    return dist

end