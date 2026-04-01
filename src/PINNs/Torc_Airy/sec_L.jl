# Seção transversal em L

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

# Define a geometria da seção transversal
function Geometria_L()

    # Lados
    a = 0.1  # [m]
    b = 0.01 # [m]

    # Coordenadas de offset da origem
    # off_x = off_y = 0 => Sem offset
    off_x = off_y = 0.0

    # Retorna os valores
    return a, b, off_x, off_y

end