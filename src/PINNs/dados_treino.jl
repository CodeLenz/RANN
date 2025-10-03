# Struct Dados_Treino
# Contém os dados para treino da rede neural: entradas e saídas
struct Treino

    # Ponto de condição inicial
    t_inicial::Vector{Float64}

    # Primeira condição incial - deslocamento
    u_inicial::Vector{Float64}

    # Segunda condição inicial - velocidade
    du_inicial::Vector{Float64}

    # Pontos de perda física
    t_fisica::Matrix{Float64}

    # Pontos de teste
    t_teste::Matrix{Float64}

    # Deslocamento analítico nos pontos de teste
    u_an::Matrix{Float64}

    # Constante de amortecimento
    μ::Float64

    # Rigidez
    k::Float64

    # Massa
    m::Float64
    
    # Função que inicializa todas as variáveis na struct
    function Treino(m::Float64, δ::Float64, ω0::Float64)

        # Verifica se o sistema é subamortecido 
        @assert δ < ω0

        # Define as constantes do problema
        # Frequência amortecida
        ω = sqrt(ω0^2 - δ^2)

        # Ângulo de fase
        ϕ = atan(- δ / ω)

        # Ampltitude
        A = 1.0 / (2.0 * cos(ϕ))

        # Constante de amortecimento
        μ = 2.0 * m * δ

        # Rigidez
        k = ω0^2 * m

        # Ponto de contorno essenciais 
        # u(t = 0) = 1
        # du(t = 0) = 0
        t_inicial = zeros(1)
        u_inicial = ones(1)
        du_inicial = zeros(1)

        # Pontos de perda física
        t_fisica = Matrix(collect(range(0.0, 1.0, 300))')

        # Pontos de teste
        t_teste = Matrix(collect(range(0.0, 1.0, 300))')

        # Deslocamento analítico nos pontos de teste
        u_an = Deslocamento(t_teste, δ, ω, A, ϕ)

        # Returna os dados
        new(t_inicial, u_inicial, du_inicial, t_fisica, t_teste, u_an, μ, k, m)

    end

end

# Calcula o deslocamento no tempo t, baseado nos parâmetros do sistema
function Deslocamento(t::Matrix{Float64}, δ::Float64, ω::Float64, A::Float64, ϕ::Float64)

    return exp.(- δ .* t) .* 2.0 .* A .* cos.(ϕ .+ ω .* t)

end 

