# =============================================================================
#  Camada de abstração de dispositivo
#
#  Detecta uma única vez (no carregamento) se existe uma GPU CUDA funcional.
#  Se existir, os arrays da rede vão para a GPU; caso contrário, tudo roda na
#  CPU normalmente. O resto do código só usa `to_device` e `dev_zeros`, sem
#  precisar saber em qual dispositivo está.
# =============================================================================

# true se há GPU CUDA utilizável, false caso contrário (roda na CPU)
const USE_GPU = CUDA.functional()

# Envia um array para o dispositivo ativo: `cu` na GPU, identidade na CPU
to_device(x) = USE_GPU ? cu(x) : x

# Aloca um array de zeros no dispositivo ativo
dev_zeros(::Type{T}, dims::Integer...) where {T<:AbstractFloat} =
    USE_GPU ? CUDA.zeros(T, dims...) : zeros(T, dims...)
