# Função para obtenção da primeira derivada da rede neural
function PrimeiraDerivada!(u::Function, du::Vector{Float64}, t::Float64)

    # Derivação automática
    Enzyme.autodiff(
                    Enzyme.Reverse,
                    u,
                    Duplicated(t, du)
                )

    return u

end

# Função para obtenção da segunda derivada da rede neural
function SegundaDerivada!(u::Function, du::Vector{Float64}, du2::Vector{Float64}, t::Float64)

    # Calcula a primeira derivada
    PrimeiraDerivada!(u, du, t)

    # Vetor unitário para multiplicação da matriz Hessiana
    # Obtém valor da diagonal, que é a segunda derivada
    v = ones(1)

    # Derivação automática para segunda derivada
    Enzyme.autodiff(Enzyme.Forward, 
                    PrimeiraDerivada!,
                    Enzyme.BatchDuplicated(zeros(1), (du2,)), 
                    Enzyme.BatchDuplicated(t, (v,))) 

end