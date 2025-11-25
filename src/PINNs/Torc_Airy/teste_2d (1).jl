#
# f:: Função a derivar R^n -> R
#
# x:: ponto atual 
#
# ϵ:: perturbação 
#
function DerivadasPDE(f0::Float64, f::Function, x::Vector, ϵ=1E-4)    

    # Número de variáveis de projeto 
    nvp = length(x)

    # Aloca a primeira derivada 
    D = zeros(nvp)

    # Aloca a segunda derivada 
    D2 = zeros(nvp)

    # Valor escalar do tempo, para propormos uma perturbação complexa
    h = ϵ*sqrt(im)

    # Promove x para complexo 
    xc = zeros(ComplexF64,nvp)
 
    # Copia os dados atuais para o vetor complexo
    xc .= x

    # Loop em cada dimensão  
    for i in eachindex(xc)

       # Backup de x[i]
       bx = xc[i]

       # Perturbação para frente
       xc[i] = bx + h

       # Calcula para frente
       ff = f(xc)

       # Perturba para trás
       xc[i] = bx - h

       # Calcula para trás
       ft = f(xc) 

       # Primeira derivada em relação a x[i]
       D[i] = real( (ff-ft)/(2*h) ) 
    
       # Segunda derivada em relação a x[i]
       D2[i] = real( (ff - 2*f0 + ft)/(h^2) )

       # Restaura 
       xc[i] = bx

    end #i

    # Retorna as derivadas 
    return D, D2

end

# Função para teste
f(x) = (x[1]-2)^2 + (x[2]-3)^2

# Ponto para teste
x = ones(2)

# Valor no ponto atual 
f0 = f(x)

# Chama a tranqueira
D, D2 = DerivadasPDE(f0, f, x) 

# Valida 
Da  = [2*(x[1]-2); 2*(x[2]-3)]
Da2 = [2.0 ; 2.0]

[D Da D2 Da2]