# Plan: PINN for Asymptotic Homogenization of 2D Elastic Composites

## Context

The existing codebase is a Julia PINN that solves the 2D torsion problem (scalar Airy stress function, Dirichlet BC via hard distance-function constraint, one PDE residual). We are extending it to solve the **asymptotic homogenization cell problem** for periodic elastic composites, computing the 3×3 effective stiffness tensor C^H in Voigt notation.

The three main structural differences from the torsion PINN are:
1. **Periodic BCs** are enforced by replacing the (y₁,y₂) input with a Fourier encoding φ(y) ∈ ℝ^{4N} — no hard constraint needed.
2. **Vector output**: network predicts 2 fluctuation displacements (ũ₁, ũ₂) instead of one scalar.
3. **Three separate training runs** (one per macroscopic strain mode ε̄^(k)), each contributing one column of C^H.

---

## File Organization

Create a new subfolder: `src/PINNs/Homogen/HomoPINN/`

Reused unchanged from parent directory (included via relative path):
- `../struct_rede.jl` — `Rede` struct, `IniciaXavier`, `Atualiza_pesos_bias` (fully generic)
- `../adamW.jl` — AdamW optimizer
- `../LBFGS.jl` — L-BFGS optimizer
- `../ativ.jl` — activation functions
- `../RNA.jl` — `RNA()` forward pass function (generic, no hard BC)

New files to create:
| File | Responsibility |
|---|---|
| `main_homo.jl` | Entry point; loops over 3 modes; calls optimizers; calls post-processing |
| `RNA_homo.jl` | Fourier encoding + forward pass (wraps existing `RNA()`) |
| `coloc_homo.jl` | Collocation grid generation on Y=[0,1]² |
| `material_homo.jl` | α(y), E(y), ν(y), C(y) (3×3 Voigt), ∂C/∂y₁, ∂C/∂y₂ for smooth circular inclusion |
| `stencil_homo.jl` | Evaluates ũ at 9 stencil points (C,E,W,N,S,NE,NW,SE,SW) with periodic wrap |
| `perdas_homo.jl` | Residuals r₁, r₂ (eqs 18-19); L_fisica; L_avg (eq 22) |
| `objetivo_homo.jl` | Full objective L = L_fisica + λ_avg·L_avg |
| `tensor_homo.jl` | Post-processing: dense grid evaluation → stress averaging → assemble C^H |
| `resultados_homo.jl` | Plots: ũ fields, stress fields, C^H matrix printout |

---

## Step-by-Step Implementation

### Step 1 — `coloc_homo.jl`: Collocation points

Generate a uniform N_c × N_c grid on Y = [0,1]², excluding the boundary strip of width h so all stencil points stay inside the unit cell without needing periodic wrap during training.

```julia
function ColocDominio_Homo(N_c::Int, h::Float64)
    # Returns 2 × N_c² matrix of interior points
    # y₁, y₂ ∈ [h, 1-h] uniformly
    ys = range(h, 1-h, length=N_c)
    pts = [y1; y2 for y2 in ys for y1 in ys]   # conceptual
    return pts  # 2 × N_c² Matrix{Float64}
end
```

A separate denser grid (N_e × N_e, e.g. 100×100, covering all of [0,1]²) is used for post-processing only.

---

### Step 2 — `material_homo.jl`: Material model

Smooth circular inclusion using tanh interpolation (eq. 14 from rodrigo_assintotico.pdf):

```
α(y) = 0.5 * (1 + tanh((R - ‖y - y_c‖) / δ))
E(y) = E₂ + (E₁ - E₂) * α(y)
ν(y) = ν₂ + (ν₁ - ν₂) * α(y)
```

Constitutive matrix in Voigt notation (eq. 13, plane stress):
```
C(y) = E(y)/(1-ν(y)²) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
```

Material derivatives (needed in equilibrium residuals eqs. 18–19) computed analytically:
```
dα/dy₁ = -0.5/δ * (1 - tanh²(·)) * (y₁ - y₁_c) / ‖y - y_c‖
dE/dy₁ = (E₁ - E₂) * dα/dy₁
dν/dy₁ = (ν₁ - ν₂) * dα/dy₁
dC/dy₁ = chain rule through E(y), ν(y), and the Voigt matrix
```

Functions to implement:
- `alpha_field(y, R, y_c, delta)` → scalar
- `material_field(y, params)` → (E, ν)
- `C_matrix(y, params)` → 3×3 SMatrix
- `dC_dy(y, params)` → (dC_dy1::3×3, dC_dy2::3×3)

Material parameters struct:
```julia
struct MaterialParams
    E1::Float64; E2::Float64
    nu1::Float64; nu2::Float64
    R::Float64; y_c::Vector{Float64}; delta::Float64
end
```

---

### Step 3 — `RNA_homo.jl`: Fourier encoding + forward pass

The Fourier encoding (eq. 5, rodrigo_assintotico.pdf) replaces the raw (y₁,y₂) input:

```julia
function fourier_encoding(y::AbstractVector{T}, N_fourier::Int)::Vector{T} where T
    # Returns vector of length 4*N_fourier
    phi = Vector{T}(undef, 4*N_fourier)
    for k in 1:N_fourier
        phi[4k-3] = sin(2π*k*y[1])
        phi[4k-2] = cos(2π*k*y[1])
        phi[4k-1] = sin(2π*k*y[2])
        phi[4k  ] = cos(2π*k*y[2])
    end
    return phi
end
```

Forward pass for homogenization:
```julia
function RNA_homo(rede, pesos, bias, y::AbstractVector{T}, N_fourier::Int)::Vector{T} where T
    phi = fourier_encoding(y, N_fourier)
    return RNA(rede, pesos, bias, phi, "Homo")   # reuses existing RNA()
end
```

Network topology: `[4*N_fourier, 50, 50, 50, 2]` (default N_fourier=4 → 16 inputs, 2 outputs)
Activations: `(tanh, tanh, tanh, identity)`

---

### Step 4 — `stencil_homo.jl`: 9-point stencil evaluation

For each collocation point y^C, evaluate the network (just forward passes) at the 9 stencil positions and return ũ at each. Periodic wrap not needed if collocation points are kept at distance ≥ h from boundary (see Step 1), so stencil coordinates are simply y ± h.

```julia
function stencil_u_tilde(rede, pesos, bias, y_C::Vector{Float64}, h::Float64, N_fourier::Int)
    y1, y2 = y_C[1], y_C[2]
    stencil_pts = [
        [y1,   y2  ],  # C
        [y1+h, y2  ],  # E
        [y1-h, y2  ],  # W
        [y1,   y2+h],  # N
        [y1,   y2-h],  # S
        [y1+h, y2+h],  # NE
        [y1-h, y2+h],  # NW
        [y1+h, y2-h],  # SE
        [y1-h, y2-h],  # SW
    ]
    # Evaluate network at each; each returns [ũ₁, ũ₂]
    return [RNA_homo(rede, pesos, bias, pt, N_fourier) for pt in stencil_pts]
end
```

Labels: C=1, E=2, W=3, N=4, S=5, NE=6, NW=7, SE=8, SW=9.

---

### Step 5 — `perdas_homo.jl`: Loss computation

**Strains** (eq. 15 from rodrigo_assintotico.pdf) using total displacements uᵢ = ε̄ᵢⱼ yⱼ + ũᵢ. Because the macroscopic part is linear, its contribution to FD formulas is exact:

```
ε₁₁ = ε̄₁₁ + (ũ₁^E - ũ₁^W) / (2h)
ε₂₂ = ε̄₂₂ + (ũ₂^N - ũ₂^S) / (2h)
ε₁₂ = ε̄₁₂ + 0.5 * ((ũ₁^N - ũ₁^S) + (ũ₂^E - ũ₂^W)) / (2h)
```

**Second derivatives** (eqs. 16–17):
```
∂²u₁/∂y₁²    = (ũ₁^E - 2ũ₁^C + ũ₁^W) / h²
∂²u₂/∂y₂²    = (ũ₂^N - 2ũ₂^C + ũ₂^S) / h²
∂²u₁/∂y₂²    = (ũ₁^N - 2ũ₁^C + ũ₁^S) / h²
∂²u₂/∂y₁²    = (ũ₂^E - 2ũ₂^C + ũ₂^W) / h²
∂²u₁/∂y₁∂y₂  = (ũ₁^NE - ũ₁^NW - ũ₁^SE + ũ₁^SW) / (4h²)
∂²u₂/∂y₁∂y₂  = (ũ₂^NE - ũ₂^NW - ũ₂^SE + ũ₂^SW) / (4h²)
```

**Equilibrium residuals** (eqs. 18–19), with Cᵢⱼ = C(y^C)[i,j] in Voigt notation
and ∂Cᵢⱼ/∂yₖ from `dC_dy`:

```
r₁ = (∂C₁₁/∂y₁)ε₁₁ + (∂C₁₂/∂y₁)ε₂₂ + (∂C₃₃/∂y₂)(2ε₁₂)
     + C₁₁(∂²u₁/∂y₁²) + C₁₂(∂²u₂/∂y₁∂y₂) + C₃₃(∂²u₁/∂y₂² + ∂²u₂/∂y₁∂y₂)

r₂ = (∂C₃₃/∂y₁)(2ε₁₂) + (∂C₂₁/∂y₂)ε₁₁ + (∂C₂₂/∂y₂)ε₂₂
     + C₃₃(∂²u₁/∂y₁∂y₂ + ∂²u₂/∂y₁²) + C₂₁(∂²u₁/∂y₁∂y₂) + C₂₂(∂²u₂/∂y₂²)
```

**Loss terms:**
```
L_fisica = (1/N_c²) * Σ (r₁² + r₂²)           # eq. 20
L_avg    = (mean(ũ₁))² + (mean(ũ₂))²           # eq. 22 (zero-mean constraint)
L_total  = L_fisica + λ_avg * L_avg             # eq. 11 from rodrigo_homo.pdf
```

---

### Step 6 — `objetivo_homo.jl`: Objective function

```julia
function Objetivo_Homo(rede::Rede, coloc::Matrix{Float64}, x::Vector{Float64},
                       epsilon_bar::Matrix{Float64}, mat::MaterialParams,
                       N_fourier::Int, h::Float64, lambda_avg::Float64)::Float64

    pesos, bias = Atualiza_pesos_bias(rede, x)
    N_c2 = size(coloc, 2)

    perda_fisica = 0.0
    sum_u1 = 0.0; sum_u2 = 0.0

    for p in 1:N_c2
        y_C = coloc[:, p]
        # 1. Evaluate ũ at 9 stencil points
        u_stencil = stencil_u_tilde(rede, pesos, bias, y_C, h, N_fourier)
        # 2. Compute strains and 2nd derivatives
        # 3. Get C(y_C), dC/dy at center point
        # 4. Compute r₁, r₂
        # 5. Accumulate loss
        perda_fisica += r1^2 + r2^2
        sum_u1 += u_stencil[1][1]   # ũ₁ at center
        sum_u2 += u_stencil[1][2]   # ũ₂ at center
    end

    perda_fisica /= N_c2
    L_avg = (sum_u1/N_c2)^2 + (sum_u2/N_c2)^2

    return perda_fisica + lambda_avg * L_avg
end
```

Enzyme computes ∂L/∂x (same pattern as existing `grad_fn!` in `main.jl`).

---

### Step 7 — `tensor_homo.jl`: Compute C^H

After training mode k, evaluate on a dense N_e × N_e grid (e.g. 100×100, covering all of [0,1]²):

```julia
function compute_CH_column(rede, x, epsilon_bar, mat, N_fourier, N_e)
    # Dense grid on [0,1]²
    # For each point y: compute ε^(k)(y) = ε̄^(k) + strain from ũ
    # Compute σ^(k)(y) = C(y) * ε^(k)(y)   (3-vector in Voigt)
    # Average σ over all grid points → one column of C^H
    # Returns [<σ₁₁>, <σ₂₂>, <σ₁₂>]
end
```

For strains on the dense grid: use the same stencil FD approach (or alternatively use the analytic Fourier derivatives since φ is smooth). Assemble:

```
C^H = [col1 | col2 | col3]    (3×3 symmetric matrix)
```

Verify symmetry: C^H[1,2] ≈ C^H[2,1], etc.

---

### Step 8 — `main_homo.jl`: Entry point

```julia
function roda_homo()
    # Problem definition
    mat = MaterialParams(E1=10.0, E2=1.0, nu1=0.3, nu2=0.3, R=0.25, y_c=[0.5,0.5], delta=0.05)
    N_fourier = 4        # 16-dim input
    N_c = 20             # 20×20 = 400 collocation points
    h = 0.5/N_c          # stencil spacing < spacing between collocation points
    lambda_avg = 10.0
    topologia = [4*N_fourier; 50; 50; 50; 2]
    ativ = (tanh, tanh, tanh, identity)

    # Generate collocation points (once, shared across modes)
    coloc = ColocDominio_Homo(N_c, h)

    # Macroscopic strain modes (Voigt: [ε₁₁, ε₂₂, ε₁₂])
    epsilon_bar_list = [
        [1.0 0.0; 0.0 0.0],   # Mode 1: extension X
        [0.0 0.0; 0.0 1.0],   # Mode 2: extension Y
        [0.0 0.5; 0.5 0.0],   # Mode 3: shear
    ]

    CH = zeros(3, 3)

    for (k, epsilon_bar) in enumerate(epsilon_bar_list)
        println("=== Treinando modo $k ===")
        rede = Rede(topologia, ativ)
        x = copy(rede.x)

        obj_fn(x) = Objetivo_Homo(rede, coloc, x, epsilon_bar, mat, N_fourier, h, lambda_avg)
        grad_fn!(G, x) = ...  # Enzyme.autodiff pattern

        x, _ = AdamW(obj_fn, grad_fn!, x, rede, ..., nepoch_ADAM)
        x, _ = LBFGS(obj_fn, grad_fn!, x, rede, ..., nepoch_LBFGS)

        CH[:, k] = compute_CH_column(rede, x, epsilon_bar, mat, N_fourier, N_e=100)
    end

    println("C^H =", CH)
    return CH
end
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Periodic BC enforcement | Fourier input encoding (eq. 5) | Exact hard enforcement; no penalty term; established pattern from literature |
| Derivative method for residuals | 9-point FD stencil on ũ | Consistent with description in rodrigo_assintotico.pdf §3.1; efficient (9 forward passes); avoids second-order Enzyme AD |
| Material property derivatives ∂C/∂y | Analytical via chain rule | Exact; smooth tanh α(y) has closed-form gradient |
| BC for rigid body modes | L_avg loss term (eq. 22) | Eliminates translation; rotation already excluded by Fourier encoding (periodic) |
| Stencil spacing h | h < 1/(2*N_c), e.g. h=0.5/N_c | Avoids stencil going outside [0,1]²; smaller than grid spacing |
| Collocation points | N_c×N_c grid, interior only (distance ≥ h from boundary) | Avoids periodic-wrap correction for macroscopic linear part |
| Post-processing grid | 100×100 independent of training grid | More accurate stress average; stated as important in rodrigo_homo.pdf |
| Optimizer | AdamW → L-BFGS (same as torsion) | Proven pattern in existing code |

---

## Verification Plan

1. **Symmetry check**: C^H should be symmetric (C^H[i,j] ≈ C^H[j,i]) — a necessary condition.
2. **Bounds check**: For any composite, Reuss bound ≤ C^H ≤ Voigt bound (component-wise for diagonal entries). Compute both analytically from E₁, E₂, ν₁, ν₂, and volume fraction f = π*R².
3. **Homogeneous limit**: Set E₁ = E₂, ν₁ = ν₂ → C^H should equal the isotropic C of that single material.
4. **Known analytical case**: For a laminate (alternating layers), C^H has a known closed form. Test with a laminate geometry (α(y) based on y₁ only, step function) → compare.
5. **Literature benchmark**: Wu et al. (2023) "Deep homogenization networks" provides C^H for a circular inclusion in a square cell with E₁/E₂ = 10, ν = 0.3, volume fraction ~20% — reproduce this case.
6. **Loss convergence**: L_fisica and L_avg should decrease monotonically and reach < 1e-5 for a well-trained network.

---

## Critical Files to Modify/Create

- Create: `HomoPINN/main_homo.jl`
- Create: `HomoPINN/RNA_homo.jl`
- Create: `HomoPINN/coloc_homo.jl`
- Create: `HomoPINN/material_homo.jl`
- Create: `HomoPINN/stencil_homo.jl`
- Create: `HomoPINN/perdas_homo.jl`
- Create: `HomoPINN/objetivo_homo.jl`
- Create: `HomoPINN/tensor_homo.jl`
- Create: `HomoPINN/resultados_homo.jl`
- Reuse (include via `../`): `struct_rede.jl`, `RNA.jl`, `adamW.jl`, `LBFGS.jl`, `ativ.jl`
