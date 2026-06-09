# Define a equação diferencial do problema e calcula o valor do resíduo no ponto
function EqDiff(Φ::Vector{Float64}, dΦ::Vector{Vector{Float64}}, dΦ2::Vector{Vector{Float64}},
                x::Vector{Float64})

    # Retorna o valor do resíduo no ponto
    return dΦ2[1][1] + dΦ2[2][1] + 2.0

end

# Calcula a resposta analítica do problema
function Φ_Analitico(prob::String, XY_teste::Matrix{Float64})

    # Caso 1: Seção transversal circular
    if prob == "Circular"

        # Aloca a resposta analítica
        Φ_analitico = zeros(1, size(XY_teste, 2))

        # Importa os dados da seção
        R, a, b, _ = Geometria_Circular()

        # Loop pelos dados de teste
        for i = 1:size(XY_teste, 2)

            # Extrai o ponto para teste
            x = treino.teste[1, i]
            y = treino.teste[2, i]
            
            # Calcula a resposta analítica
            Φ_analitico[1, i] = 0.5 * (R^2 - (x - a)^2 - (y - b)^2)
            
        end

        # Retorna resposta
        return Φ_analitico

    # Caso 2: Seção transversal retangular
    elseif prob == "Retangular"

        # Aloca a resposta analítica
        Φ_analitico = zeros(1, size(XY_teste, 2))

        # Importa os dados da seção
        H, B, a, b, _ = Geometria_Retangular()

        # Loop pelos dados de teste
        for i = 1:size(XY_teste, 2)

            # Extrai o ponto para teste
            x = treino.teste[1, i]
            y = treino.teste[2, i]
            
            # Loop por N termos da série, a princípio deixaremos 20
            N = 20

            # Acumula
            Φ = 0.0
            for ki in 0:(N-1)

                k = 2 * ki + 1

                for li in 0:(N-1)

                    l = 2 * li + 1

                    num = (-1)^( ( (k + l) ÷ 2 ) - 1 )

                    Φ += ( num / (k * l * (k^2 * H^2 + l^2 * B^2)) ) * cos(k * π * x / B) * cos(l * π * y / H)

                end

            end

            # Solução analítica
            Φ_analitico[1, i] = Φ * (32 * B^2 * H^2 / π^4)
            
        end

        # Retorna resposta
        return Φ_analitico

    # Caso 3: Seção transversal em L
    elseif prob == "L"

        # Importa os dados da seção
        a, b, off_x, off_y = Geometria_L()

        # Lê os dados diretamente do arquivo de resultados nodais de FEM
        solucao_nodal = readdlm("solucao_airy_nodal.txt")

        # Filtra posições do contorno
        indices_contorno = findall(λ -> !ponto_no_contorno(solucao_nodal[λ, 1], solucao_nodal[λ, 2], a, b), 1:size(solucao_nodal, 1))
        solucao_nodal = solucao_nodal[indices_contorno, :]

        # Seleciona somente os valores de Φ
        Φ_analitico = Matrix(transpose(solucao_nodal[:, 4]))

        # Retorna resposta
        return Φ_analitico

    # Caso não seja selecionado nenhum problema, retorna vazio
    else

        return Vector{Float64}()

    end

end