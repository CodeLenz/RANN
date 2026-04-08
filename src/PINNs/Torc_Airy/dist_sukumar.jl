# Função de distância do contorno para aplicação forte das condições de contorno
# Proposta retirada de N. Sukumar, Ankit Srivastava
# https://arxiv.org/abs/2104.08426

# ADF para retângulo [-H/2, H/2] × [-B/2, B/2]
# Segmentos seguem ordem anti-horária: baixo, direita, topo, esquerda
function Distancia_Contorno_Retangular(XY::AbstractVector{T}) where T

    # Para facilitar 
    x = XY[1]
    y = XY[2]

    # Importa os dados da seção
    H, B, a, b, _ = Geometria_Retangular()

    # Quatro segmentos do retângulo, sentido anti-horário
    # x, y, x1, y1, x2, y2, L
    φ1 = adf_segmento(x, y, -B/2, -H/2,  B/2, -H/2, B)  # baixo
    φ2 = adf_segmento(x, y,  B/2, -H/2,  B/2,  H/2, H)  # direita
    φ3 = adf_segmento(x, y,  B/2,  H/2, -B/2,  H/2, B)  # topo
    φ4 = adf_segmento(x, y, -B/2,  H/2, -B/2, -H/2, H)  # esquerda

    # Computa a função ADF equivalente combinando os quatro segmentos
    φ_equiv = adf_requivalente([φ1, φ2, φ3, φ4])

    # Combina os segmentos usando R-equivalence joining
    return φ_equiv

end

# ADF para seção em L [0, a] × [0, b]
# Segmentos seguem ordem anti-horária
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
function Distancia_Contorno_L(XY::AbstractVector{T}) where T

    # Para facilitar 
    x = XY[1]
    y = XY[2]

    # Importa os dados da seção
    a, b, off_x, off_y = Geometria_L()

    # Seis segmentos do L, sentido anti-horário
    # x, y, x1, y1, x2, y2, L
    φ1 = adf_segmento(x, y, 0.0, 0.0, a, 0.0, a)  
    φ2 = adf_segmento(x, y, a, 0.0, a, b, b) 
    φ3 = adf_segmento(x, y, a, b, b, b, (a-b)) 
    φ4 = adf_segmento(x, y, b, b, b, a, (a-b))
    φ5 = adf_segmento(x, y, b, a, 0.0, a, b)
    φ6 = adf_segmento(x, y, 0.0, a, 0.0, 0.0, a)  

    # Computa a função ADF equivalente combinando os quatro segmentos
    φ_equiv = adf_requivalente([φ1, φ2, φ3, φ4, φ5, φ6])

    # Combina os segmentos usando R-equivalence joining
    return φ_equiv

end

# Calcula a função ADF (Approximate Distance Function) para um segmento de linha
# Equação 6 no artigo
# x1,y1 e x2,y2 são as extremidades do segmento, x,y é o ponto onde a função é avaliada
function adf_segmento(x::T, y::T, x1::Float64, y1::Float64, x2::Float64, y2::Float64, L::Float64) where T

    # Centro do segmento
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    # Distância para linha infinita (equação 4)
    f = ((x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)) / L

    # Função de trimming — positiva dentro de disco de raio L/2 (equação 5)
    t = (1/L) * ((L/2)^2 - ((x - xc)^2 + (y - yc)^2))

    # ADF ao segmento (equação 6)
    φ_til = sqrt(t^2 + f^4)
    φ = sqrt(f^2 + ((φ_til - t)/2)^2)

    return φ

end

# Combina n segmentos usando R-equivalence joining (equação 10, m=1)
function adf_requivalente(vetor_φ::AbstractVector{T}; m = 2) where T

    # φ = 1 / (Σ 1/φᵢᵐ)^(1/m)
    denominador = sum(1.0 / φ^m for φ in vetor_φ)

    # Retorna φ equivalente
    return (1.0 / denominador)^(1/m)
end

