
using LinearAlgebra
using Enzyme
using Random; Random.seed!(42)

#
# Cálculo do deslocamento usando a rede neural, para um dado conjunto 
# de parâmetros x da rede e um tempo t
#
function resposta(x::Vector,t::Vector)

    ctes = [1.0; 2.0; 3.0; 4.0; 6.0]
    (t[1]^2) * dot(ctes, x)

end

function dresposta(x::Vector,t::Vector)

    ctes = [1.0; 2.0; 3.0; 4.0; 6.0]
    (2*t[1]) * dot(ctes, x)

end

function d2resposta(x::Vector,t::Vector)

    ctes = [1.0; 2.0; 3.0; 4.0; 6.0]
    (2) * dot(ctes, x)

end

#
# Calcula o resíduo da EDO
#
#  r(x, t) =  mü(x, t) + cu̇(x, t) + ku(x, t) - f(x, t) 
#
# em que x é o vetor com os parâmetros da rede e t é o tempo
# variável independente da EDO
#
function residuo(x::Vector, tempo::Vector{Float64})::Float64

    # Dados da EDO
    m = 1.0
    k = 10.0
    ζ  = 1/100
    c = ζ*2*sqrt(k*m)
    f(t) = 10*cos(t)

    # Derivadas da função u
    u,du,du2 = Derivadas(resposta,tempo,x)
  
    # O resíduo no tempo t, para a rede x será 
    r = m*du2[1] + c*du[1] + k*u - f(tempo[1])
     
end

#
# Calcula a resposta, a primeira derivada e a segunda derivada em 
# relação ao tempo para um conjunto x de parâmetros fixos da rede
#
function Derivadas(u::Function,tempo::Vector{Float64},x::Vector)
    
    # Aloca as derivadas
    du = zeros(1)
    du2 = zeros(1)

    # Driver para calcular a derivada em relação a t
    function derivada!(x::Vector,saida::Vector,ponto::Vector)
             Enzyme.autodiff(
                        Enzyme.set_runtime_activity(Enzyme.Reverse),
                        u,
                        Enzyme.Const(x),
                        Enzyme.Duplicated(ponto, saida)
                    )
    end
   
    # Calcula a primeira derivada
    derivada!(x,du,tempo)

    # Derivada para frente
    ddu = zeros(1)
    derivada!(x,ddu,tempo.+1E-6)
    du2 = (ddu.-du)./1E-6

    @show du,dresposta(x,tempo)
    @show du2,d2resposta(x,tempo)

    

    # Multiplica a Hessiana pelo vetor [1]
    # v = ones(1)

    # Aloca a saída
    #du2 = zeros(1)

    # Calcula a segunda derivada em relação ao tempo
    #=
    Enzyme.autodiff(Enzyme.Forward, 
                    derivada!,
                    Enzyme.Const(x),
                    Enzyme.BatchDuplicated(zeros(1), (du2,)), 
                    Enzyme.BatchDuplicated(tempo, (v,))) 
    =#

    return u(x,tempo), du, du2

end

 # Calcula os resíduos
 x = [1.0;2.0;3.0;4.0;5.0]
 r = [residuo(x,[t]) for t in 0:0.1:1.0]
 norm(r)

 #
 # O problema que estamos tendo é derivada em relação a x (parâmetros da rede)
 # considerando que já estamos utilizando o Enzyme para calcular as derivadas 
 # da EDO
 dx = similar(x)
 ponto = [1.0]
 dponto = [1.0]
 Enzyme.autodiff(
                        Enzyme.set_runtime_activity(Enzyme.Reverse),
                        residuo,
                        Duplicated(x,dx),
                        Enzyme.Const(ponto)
                    )


#
# Valida por DF
#
xc = copy(x)

# Referência para o resíduo
r0 = residuo(x,[1.0]) 

D = zeros(5) 

for i=1:5

    xb = xc[i]
    xc[i] += 1E-6
    rf = residuo(xc,[1.0])
    D[i] = (rf-r0)/1E-6
    xc[i] = xb

end