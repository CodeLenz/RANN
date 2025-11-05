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

    # Intensidade da força
    F::Float64

    # Frequência da força
    ωf::Float64
    
    # Função que inicializa todas as variáveis na struct
    function Treino(m::Float64, ζ::Float64, ω0::Float64, F::Float64, ωf::Float64)

        # Define as constantes do problema
        # Razão de frequências
        r = ωf / ω0

        # Rigidez
        k = ω0^2 * m

        # Ângulo de fase
        ϕ = atan(2 * ζ * r, 1 - r^2)

        # Amplitude em regime permanente
        X = (F / k) / sqrt((1 - r^2)^2 + (2 * ζ * r)^2)

        # Frequência amortecida
        ωd = ω0 * sqrt(1 - ζ^2) 

        # Constantes A e B da solução
        A = - X * cos(ϕ)
        B = (X / ωd) * (ωf * sin(ϕ) - ζ * ω0 * cos(ϕ))

        # Constante de amortecimento
        μ = 2.0 * m * ζ * ω0

        # Ponto de contorno essenciais 
        # u(t = 0) = 0
        # du(t = 0) = 0
        t_inicial = zeros(1)
        u_inicial = zeros(1)
        du_inicial = zeros(1)

        # Pontos de perda física
        t_fisica = Matrix(collect(range(0.0, 1.0, 50))')

        # Pontos de teste
        t_teste = Matrix(collect(range(0.0, 1.0, 300))')

        # Deslocamento analítico nos pontos de teste
        u_an = Deslocamento(t_teste, ζ, ω0, ωd, ωf, ϕ, X, A, B)

        # Returna os dados
        new(t_inicial, u_inicial, du_inicial, t_fisica, t_teste, u_an, μ, k, m, F, ωf)

    end

end

# Calcula o deslocamento no tempo t, baseado nos parâmetros do sistema
function Deslocamento(t::Matrix{Float64}, ζ::Float64, ω0::Float64, ωd::Float64, ωf::Float64,
                      ϕ::Float64, X::Float64, A::Float64, B::Float64)

    return exp.(-ζ .* ω0 .* t) .* (A .* cos.(ωd .* t) + B .* sin.(ωd .* t)) .+ X .* cos.(ωf .* t .- ϕ)


end 

