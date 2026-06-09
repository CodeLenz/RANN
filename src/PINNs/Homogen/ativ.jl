# -----------------------------------------------------------------------------
#  Funções de ativação parametrizadas para a saída ativada 'a'
# -----------------------------------------------------------------------------
TANH_GEN   = (tanh,     a -> one(a) - a^2)
LINEAR_GEN = (identity, a -> one(a))