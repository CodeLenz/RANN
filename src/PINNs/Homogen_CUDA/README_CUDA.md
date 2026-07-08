# Homogen — versão GPU (CUDA) / device-agnostic

Cópia da pasta `Homogen` adaptada para rodar na GPU com **CUDA.jl**.
As alterações foram mínimas: os *arrays* da rede, os *buffers* de ativação e a
matriz de colocação passam a viver na GPU (`CuArray`); a geração de pontos
(Sobol), as propriedades do material e a montagem final do tensor continuam na
CPU (trabalho escalar e barato).

**Device-agnostic:** o código detecta automaticamente se há uma GPU CUDA
funcional (`CUDA.functional()`, em `device.jl`). Se houver, roda na GPU; se não
houver, roda **na CPU** normalmente — sem alterar nada. Toda a transferência
passa por `to_device(...)` e `dev_zeros(...)`, que viram identidade/`zeros` na
CPU. (Verificado: o caminho de CPU roda o pipeline completo de ponta a ponta.)

## Pré-requisito

Uma GPU NVIDIA + driver CUDA e o pacote CUDA.jl no ambiente:

```julia
using Pkg
Pkg.activate("caminho/para/RANN")   # o mesmo Project.toml usado pelo original
Pkg.add("CUDA")
```

## Como rodar

```julia
include("main.jl")   # a partir desta pasta (Homogen_CUDA)
```

Os resultados são gravados em `Resultados/` (criada vazia aqui, para não
misturar com os resultados da versão CPU).

## O que mudou em relação à versão CPU (resumo)

| Arquivo          | Mudança |
|------------------|---------|
| `device.jl`      | **novo**: `USE_GPU`, `to_device(...)`, `dev_zeros(...)` — escolhem GPU ou CPU |
| `main.jl`        | `using CUDA`; inclui `device.jl`; escolhe device (GPU/CPU) sem erro; leitura de parâmetros via `to_device(...)`; `ENV["GKSwstype"]=100` (plot headless) |
| `RNA.jl`         | campos de `Camada` viram `AbstractMatrix/AbstractVector`; init dos pesos com `to_device(...)`; assinaturas de `Forward`/`Backward` relaxadas para `AbstractMatrix`/`AbstractVector` |
| `PINN.jl`        | `Perda_Energia_Alvo` aceita `AbstractMatrix`; bloco do material embrulhado em `Zygote.@ignore` e enviado ao device com `to_device(...)` |
| `treino.jl`      | `X_all`, `As`, `Z_buffers` via `to_device` / `dev_zeros` |
| `AdamW.jl`       | momentos inicializados com `zero(c.W)`/`zero(c.b)` (mesmo *device* dos pesos) |
| `tensao_pos.jl`  | *buffers* via `dev_zeros`; deformações voltam para a CPU (`Array`) antes da montagem ponto a ponto |
| `resultados.jl`  | `writedlm` recebe `Array(c.W)`/`Array(c.b)` |
| `L-BFGS/*`       | tipo da fila de memória e assinaturas de `LS_Wolfe`/`Refinamento` relaxados para `AbstractVector`; `r = zero(q)` |

A lógica, os nomes e os comentários do algoritmo permanecem os mesmos.
