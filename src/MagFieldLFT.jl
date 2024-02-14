module MagFieldLFT

using LinearAlgebra, Permutations, OutputParser, DelimitedFiles, Printf, TensorOperations

export read_AILFT_params_ORCA, LFTParam, lebedev_grids

const kB = 3.166811563e-6    # Boltzmann constant in Eh/K
const alpha = 0.0072973525693  # fine structure constant
const mu0 = 4pi*alpha^2

function xyz2spher(x::Real, y::Real, z::Real)
    theta = acos(z)
    phi = atan(y,x)   # 2-argument atan (also known as atan2)
    return theta, phi
end

function setup_Lebedev_grids()
    pkgpath = dirname(pathof(MagFieldLFT))
    gridsizes = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810]
    lebedev_grids = Vector{Vector{Tuple{Float64, Float64, Float64}}}(undef, 0)
    for N in gridsizes
        xyzgrid = readdlm("$pkgpath/grids/grid_$N")
        grid = Vector{Tuple{Float64, Float64, Float64}}(undef, 0)
        for i in 1:(size(xyzgrid)[1])
            theta, phi = xyz2spher(xyzgrid[i,1], xyzgrid[i,2], xyzgrid[i,3])
            weight = 4pi * xyzgrid[i,4]
            push!(grid, (theta, phi, weight))
        end
        push!(lebedev_grids, grid)
    end
    return lebedev_grids
end

lebedev_grids = setup_Lebedev_grids()

"""
nel: Number of electrons
norb: Number of orbitals (=2l+1)
l: Angular momentum of the open shell (l=2 for d orbitals, l=3 for f orbitals)
hLFT: One-electron ligand field matrix
F: Racah parameters
zeta: Spin-orbit coupling parameter
"""
struct LFTParam
    nel::Int64
    norb::Int64
    l::Int64
    hLFT::Matrix{Float64}
    F::Dict{Int64, Float64}
    zeta::Float64
end

function create_SDs(nel::Int, norb::Int)
    nspinorb = 2*norb
    SDs = [[P] for P in 1:(nspinorb-nel+1)]
    if length(SDs[1])<nel
        construct_SDs_recursive!(SDs, nspinorb, nel)
    end
    return SDs
end

function construct_SDs_recursive!(SDs::Vector{Vector{Int64}}, nspinorb::Int, nel::Int)
    oldSDs = deepcopy(SDs)
    deleteat!(SDs, 1:length(SDs))
    for SD in oldSDs
        for P in (SD[end]+1):nspinorb
            newSD = deepcopy(SD)
            push!(newSD, P)
            push!(SDs, newSD)
        end
    end
    if length(SDs[1])<nel
        construct_SDs_recursive!(SDs, nspinorb, nel)
    end
    return nothing
end

function U_complex2real(l::Int)
    dim = 2l+1
    U = im*zeros(dim,dim)
    mvalues = l:-1:-l
    for i in 1:dim
        m = mvalues[i]
        if m>0
            U[i,i] = (-1)^m / sqrt(2)
            U[dim+1-i,i] = 1/sqrt(2)
        elseif m==0
            U[i,i] = 1.0
        else
            U[i,i] = im/sqrt(2)
            U[dim+1-i,i] = -(-1)^m * im/sqrt(2)
        end
    end
    return U
end

c1_0 = Matrix(1.0I, 3, 3)
c1_2 = [-1 sqrt(3) -sqrt(6);
        -sqrt(3) 2 -sqrt(3);
        -sqrt(6) sqrt(3) -1]

c2_0 = Matrix(1.0I, 5, 5)
c2_2 = [-2 sqrt(6) -2 0 0;
        -sqrt(6) 1 1 -sqrt(6) 0;
        -2 -1 2 -1 -2;
        0 -sqrt(6) 1 1 -sqrt(6);
        0 0 -2 sqrt(6) -2]
c2_4 = [1 -sqrt(5) sqrt(15) -sqrt(35) sqrt(70);
        sqrt(5) -4 sqrt(30) -sqrt(40) sqrt(35);
        sqrt(15) -sqrt(30) 6 -sqrt(30) sqrt(15);
        sqrt(35) -sqrt(40) sqrt(30) -4 sqrt(5);
        sqrt(70) -sqrt(35) sqrt(15) -sqrt(5) 1]

c3_0 = Matrix(1.0I, 7, 7)
c3_2 = [-5 5 -sqrt(10) 0 0 0 0;
        -5 0 sqrt(15) -sqrt(20) 0 0 0;
        -sqrt(10) -sqrt(15) 3 sqrt(2) -sqrt(24) 0 0;
        0 -sqrt(20) -sqrt(2) 4 -sqrt(2) -sqrt(20) 0;
        0 0 -sqrt(24) sqrt(2) 3 -sqrt(15) -sqrt(10);
        0 0 0 -sqrt(20) sqrt(15) 0 -5;
        0 0 0 0 -sqrt(10) 5 -5]
c3_4 = [3 -sqrt(30) sqrt(54) -sqrt(63) sqrt(42) 0 0;
        sqrt(30) -7 4*sqrt(2) -sqrt(3) -sqrt(14) sqrt(70) 0;
        sqrt(54) -4*sqrt(2) 1 sqrt(15) -sqrt(40) sqrt(14) sqrt(42);
        sqrt(63) -sqrt(3) -sqrt(15) 6 -sqrt(15) -sqrt(3) sqrt(63);
        sqrt(42) sqrt(14) -sqrt(40) sqrt(15) 1 -4*sqrt(2) sqrt(54);
        0 sqrt(70) -sqrt(14) -sqrt(3) 4*sqrt(2) -7 sqrt(30);
        0 0 sqrt(42) -sqrt(63) sqrt(54) -sqrt(30) 3]
c3_6 = [-1 sqrt(7) -sqrt(28) sqrt(84) -sqrt(210) sqrt(462) -sqrt(924);
        -sqrt(7) 6 -sqrt(105) 4*sqrt(14) -sqrt(378) sqrt(504) -sqrt(462);
        -sqrt(28) sqrt(105) -15 5*sqrt(14) -sqrt(420) sqrt(378) -sqrt(210);
        -sqrt(84) 4*sqrt(14) -5*sqrt(14) 20 -5*sqrt(14) 4*sqrt(14) -sqrt(84);
        -sqrt(210) sqrt(378) -sqrt(420) 5*sqrt(14) -15 sqrt(105) -sqrt(28);
        -sqrt(462) sqrt(504) -sqrt(378) 4*sqrt(14) -sqrt(105) 6 -sqrt(7);
        -sqrt(924) sqrt(462) -sqrt(210) sqrt(84) -sqrt(28) sqrt(7) -1]

"""
Keys are tuples (l,k), where l is the angular momentum
quantum number (e.g. l=2 for d orbitals) and k the index
of the Slater--Condon parameters.
"""
c_matrices = Dict((1,0) => c1_0,
                  (1,2) => c1_2,
                  (2,0) => c2_0,
                  (2,2) => c2_2,
                  (2,4) => c2_4,
                  (3,0) => c3_0,
                  (3,2) => c3_2,
                  (3,4) => c3_4,
                  (3,6) => c3_6)

"""
Convert Racah parameters for a shell of d orbitals
into Slater--Condon parameters.
"""
function Racah2F(A::Real, B::Real, C::Real)
    return Dict(0 => A+(7/5)*C, 2 => B+(1/7)*C, 4 => C/35)
end

"""
Calculate lz operator in basis of complex atomic orbitals.
"""
calc_lz(l::Int) = diagm(l:-1:-l)

function calc_lplusminus(l::Int, sign::Int)
    @assert Int64(abs(sign)) == 1       # sign may only be +1 or -1
    dim = 2l+1
    mvalues = l:-1:-l
    op = zeros(dim,dim)
    for i_prime in 1:dim
        m_prime = mvalues[i_prime]
        for i in 1:dim
            m = mvalues[i]
            if m_prime == (m + sign)
                op[i_prime, i] = sqrt(l*(l+1) - m*(m+sign))
            end
        end
    end
    return op
end

"""
Calculate sz operator in basis of complex atomic orbitals.
"""
calc_sz(l::Float64) = diagm(l:-1:-l)

function calc_splusminus(l::Float64, sign::Int)
    @assert Int64(abs(sign)) == 1       # sign may only be +1 or -1
    dim = Int(2l+1)
    mvalues = l:-1:-l
    op = zeros(dim,dim)
    for i_prime in 1:dim
        m_prime = mvalues[i_prime]
        for i in 1:dim
            m = mvalues[i]
            if m_prime == (m + sign)
                op[i_prime, i] = sqrt(l*(l+1) - m*(m+sign))
            end
        end
    end
    return op
end

"""
Returns lz, lplus, lminus in basis of complex atomic orbitals of angular momentum l.
"""
function calc_lops_complex(l::Int)
    return calc_lz(l), calc_lplusminus(l,+1), calc_lplusminus(l,-1)
end

function calc_lops_real(l::Int)
    lz, lplus, lminus = calc_lops_complex(l)
    U = U_complex2real(l)
    return U'*lz*U, U'*lplus*U, U'*lminus*U
end

"""
We do not make use of the symmetries of ERIs since their
number is anyway small in single-shell LFT with d or f orbitals.
This simplifies the implementation.
This function calculates ERIs of complex orbitals
"""
function calcERIs_complex(l::Int, F::Dict{Int64, Float64})
    dim = 2l+1
    m = l:-1:-l
    ERIs = zeros(dim, dim, dim, dim)
    for p in 1:dim, q in 1:dim, r in 1:dim, s in 1:dim
        if m[p]-m[q] == m[s]-m[r]
            for k in 0:2:2l
                ERIs[p,q,r,s] += c_matrices[(l,k)][p,q] * c_matrices[(l,k)][s,r] * F[k]
            end
        end
    end
    return ERIs
end

function calcERIs_real(l::Int, F::Dict{Int64, Float64})
    dim = 2l+1
    U = U_complex2real(l)
    # The following loop structure has O(8) complexity instead of O(5)
    # for the ERI transformation. But we can afford it for such a
    # small number of orbitals (5 or 7), and it makes the code simpler.
    ERIs_complex = calcERIs_complex(l, F)
    ERIs_real = im*zeros(dim, dim, dim, dim)
    for p in 1:dim, q in 1:dim, r in 1:dim, s in 1:dim
        for t in 1:dim, u in 1:dim, v in 1:dim, w in 1:dim
            ERIs_real[t,u,v,w] += ERIs_complex[p,q,r,s] * U[p,t]' * U[q,u] * U[r,v]' * U[s,w]
        end
    end
    @assert norm(imag(ERIs_real)) < 1e-12
    return real(ERIs_real)
end

function spinorb2orbindex(P::Int)
    p = (P+1) ÷ 2
    if (P+1)%2 == 0
        sigma = 'α'
    else
        sigma = 'β'
    end
    return p,sigma
end

function orb2spinorbindex(p::Int, sigma::Char)
    @assert sigma == 'α' || sigma == 'β'
    if sigma == 'α'
        return 2p-1
    else
        return 2p
    end
end

"""
For a given Slater determinant (Vector of spin orbital indices in canonical order),
this function returns a list of either the alpha or the beta orbitals, depending on the
chosen channel.
For each spin orbital that belongs to the requested channel, a tuple of the position
in the Slater determinant (electron index) and the corresponding orbital index is returned.
"""
function occ_list(SD::Vector{Int64}, channel::Char)
    @assert channel == 'α' || channel == 'β'
    list = Array{Tuple{Int64,Int64}}(undef,0)
    for i in 1:length(SD)
        p,sigma = spinorb2orbindex(SD[i])
        if sigma == channel
            push!(list, (i,p))
        end
    end
    return list
end

"""
SD: List of spin orbital indices in canonical order
norb: number of spatial orbitals
channel: 'α' or 'β'
"""
function unocc_list(SD::Vector{Int64}, norb::Int, channel::Char)
    @assert channel == 'α' || channel == 'β'
    list = Array{Int64}(undef,0)
    for p in 1:norb
        P = orb2spinorbindex(p,channel)
        if !(P in SD)
            push!(list, p)
        end
    end
    return list
end

function Z_summand(i::Int64, P::Int64, N::Int64, M::Int64)
    @assert i<=N
    if i==N
        return P-N
    else
        value = 0
        for m in (M-P+1):(M-i)
            value += binomial(m, N-i) - binomial(m-1, N-i-1)
        end
        return value
    end
end

"""
SD: List of spin orbital indices
M:  Total number of spin orbitals
"""
function SD2index(SD::Vector{Int64}, M::Int64)
    N = length(SD)
    I = 1
    for i in 1:N
        I += Z_summand(i, SD[i], N, M)
    end
    return I
end

function calc_exc_occ2unocc(SD::Vector{Int64}, norb::Int, sigma_p::Char, sigma_q::Char)
    @assert sigma_p == 'α' || sigma_p == 'β'
    @assert sigma_q == 'α' || sigma_q == 'β'
    occ = occ_list(SD, sigma_q)
    unocc = unocc_list(SD, norb, sigma_p)
    exc = Array{Tuple{Int64, Int64, Int64, Int64}}(undef,0)
    for (i,q) in occ
        for p in unocc
            P = orb2spinorbindex(p, sigma_p)   # spin orbital index we are exciting to
            SD_exc = deepcopy(SD)
            SD_exc[i] = P
            perm = sortperm(SD_exc)
            SD_exc_canonical = SD_exc[perm]
            perm = Permutation(perm)
            gamma = sign(perm)
            I = SD2index(SD_exc_canonical, 2norb)
            push!(exc, (I, p, q, gamma))
        end
    end
    return exc
end

function calc_exc_occ2self(SD::Vector{Int64}, norb::Int, channel::Char)
    @assert channel == 'α' || channel == 'β'
    occ = occ_list(SD, channel)
    exc = Array{Tuple{Int64, Int64, Int64, Int64}}(undef,0)
    K = SD2index(SD, 2norb)
    for (i,q) in occ
        push!(exc, (K, q, q, 1))
    end
    return exc
end

function calc_hmod(hLFT::Matrix{Float64}, ERIs::Array{Float64, 4})
    hmod = deepcopy(hLFT)
    dim = size(hmod)[1]
    for p in 1:dim
        for q in 1:dim
            for r in 1:dim
                hmod[p,q] -= 0.5* ERIs[p,r,r,q]
            end
        end
    end
    return hmod
end

struct ExcitationLists
    alpha::Vector{Vector{NTuple{4, Int64}}}
    beta::Vector{Vector{NTuple{4, Int64}}}
    plus::Vector{Vector{NTuple{4, Int64}}}
    minus::Vector{Vector{NTuple{4, Int64}}}
    Dim::Int64
end

"""
l: angular momentum of the partially filled shell (l=2 for d orbitals, l=3 for f orbitals).
N: number of electrons in partially filled shell
"""
function calc_exclists(l::Int, N::Int)
    norb = 2l+1
    SDs = create_SDs(N, norb)
    Dim = length(SDs)
    L_alpha = Vector{Vector{NTuple{4, Int64}}}(undef, 0)
    L_beta = Vector{Vector{NTuple{4, Int64}}}(undef, 0)
    L_plus = Vector{Vector{NTuple{4, Int64}}}(undef, 0)
    L_minus = Vector{Vector{NTuple{4, Int64}}}(undef, 0)
    for K in 1:Dim
        exc_alpha_diff = calc_exc_occ2unocc(SDs[K], norb, 'α', 'α')
        exc_beta_diff = calc_exc_occ2unocc(SDs[K], norb, 'β', 'β')
        exc_plus = calc_exc_occ2unocc(SDs[K], norb, 'α', 'β')
        exc_minus = calc_exc_occ2unocc(SDs[K], norb, 'β', 'α')
        exc_alpha_same = calc_exc_occ2self(SDs[K], norb, 'α')
        exc_beta_same = calc_exc_occ2self(SDs[K], norb, 'β')
        push!(L_alpha, [exc_alpha_same; exc_alpha_diff])
        push!(L_beta, [exc_beta_same; exc_beta_diff])
        push!(L_plus, exc_plus)
        push!(L_minus, exc_minus)
    end
    Dim = length(L_alpha)
    return ExcitationLists(L_alpha, L_beta, L_plus, L_minus, Dim)
end

function calc_exclists(param::LFTParam)
    l = (param.norb-1)÷2
    return calc_exclists(l,param.nel)
end

function calc_singletop(int::Matrix{T}, exc::ExcitationLists) where T<:Number
    H_single = convert(T, 0) * zeros(exc.Dim, exc.Dim)   # zero matrix with same type as int
    for J in 1:exc.Dim
        for (Ii,pi,qi,gammai) in exc.alpha[J]
            H_single[Ii,J] += int[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in exc.beta[J]
            H_single[Ii,J] += int[pi,qi]*gammai
        end
    end
    return H_single
end

function calc_double_exc(ERIs::Array{Float64, 4}, exc::ExcitationLists)
    norb = size(ERIs)[1]
    H_double = zeros(exc.Dim, exc.Dim)
    for K in 1:exc.Dim
        X = zeros(exc.Dim, norb, norb)
        for p in 1:norb, q in 1:norb
            for (Ii,pi,qi,gammai) in exc.alpha[K]
                X[Ii,p,q] += ERIs[p,q,qi,pi] * gammai
            end
            for (Ii,pi,qi,gammai) in exc.beta[K]
                X[Ii,p,q] += ERIs[p,q,qi,pi] * gammai
            end
        end
        for J in 1:exc.Dim
            for (Ii,pi,qi,gammai) in exc.alpha[K]
                H_double[Ii, J] += 0.5*gammai*X[J,pi,qi]
            end
            for (Ii,pi,qi,gammai) in exc.beta[K]
                H_double[Ii, J] += 0.5*gammai*X[J,pi,qi]
            end
        end
    end
    return H_double
end

function calc_H_nonrel(param::LFTParam, exc::ExcitationLists)
    norb = size(param.hLFT)[1]
    l = (norb-1)÷2
    ERIs = calcERIs_real(l, param.F)
    h_mod = calc_hmod(param.hLFT, ERIs)
    H_single = calc_singletop(h_mod, exc)
    H_double = calc_double_exc(ERIs, exc)
    return H_single + H_double
end

function calc_SOCintegrals(zeta::Float64, l::Int)
    lz, lplus, lminus = calc_lops_real(l::Int)
    return zeta*lz, zeta*lplus, zeta*lminus
end

"""
zeta: SOC parameter
l: angular momentum of partially filled shell
exc: Lists of excitations and coupling coefficients.
"""
function calc_SOC(SOCints::NTuple{3, Matrix{ComplexF64}}, exc::ExcitationLists)
    H_SOC = im*zeros(exc.Dim, exc.Dim)
    SOCz, SOCplus, SOCminus = SOCints
    for J in 1:exc.Dim
        for (Ii,pi,qi,gammai) in exc.minus[J]
            H_SOC[Ii,J] += 0.5*SOCplus[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in exc.plus[J]
            H_SOC[Ii,J] += 0.5*SOCminus[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in exc.alpha[J]
            H_SOC[Ii,J] += 0.5*SOCz[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in exc.beta[J]
            H_SOC[Ii,J] -= 0.5*SOCz[pi,qi]*gammai
        end
    end
    return H_SOC
end

function calc_H_fieldfree(param::LFTParam, exc::ExcitationLists)
    norb = size(param.hLFT)[1]
    l = (norb-1)÷2
    SOCints = calc_SOCintegrals(param.zeta, l)
    H_fieldfree = calc_H_nonrel(param, exc) + calc_SOC(SOCints, exc)
    return Hermitian(H_fieldfree)
end

function calc_L(l::Int, exc::ExcitationLists)
    lz, lplus, lminus = calc_lops_real(l::Int)
    Lz = calc_singletop(lz, exc)
    Lplus = calc_singletop(lplus, exc)
    Lminus = calc_singletop(lminus, exc)
    Lx = (Lplus + Lminus)/2
    Ly = (Lplus - Lminus)/(2im)
    return Lx, Ly, Lz
end

function calc_S(l::Int, exc::ExcitationLists)
    norb = 2l+1
    delta = Matrix{Float64}(1.0I, norb, norb)
    Dim = length(exc.alpha)
    Sx = zeros(Dim, Dim)
    Sy = im*zeros(Dim, Dim)
    Sz = zeros(Dim, Dim)
    for J in 1:Dim
        for (Ii,pi,qi,gammai) in exc.plus[J]
            Sx[Ii,J] += delta[pi,qi]*gammai/2
            Sy[Ii,J] += delta[pi,qi]*gammai/(2im)
        end
        for (Ii,pi,qi,gammai) in exc.minus[J]
            Sx[Ii,J] += delta[pi,qi]*gammai/2
            Sy[Ii,J] -= delta[pi,qi]*gammai/(2im)
        end
        for (Ii,pi,qi,gammai) in exc.alpha[J]
            Sz[Ii,J] += delta[pi,qi]*gammai/2
        end
        for (Ii,pi,qi,gammai) in exc.beta[J]
            Sz[Ii,J] -= delta[pi,qi]*gammai/2
        end
    end
    return Sx, Sy, Sz
end

"""
Read parameters (in atomic units).
method: e.g. "CASSCF" or "NEVPT2"
TO DO: extend for f elements and for SOC parameter.
"""
function read_AILFT_params_ORCA(outfile::String, method::String)
    nel = parse_int(outfile, ["nel"], 0, 3)
    norb = parse_int(outfile, ["norb"], 0, 3)
    l = (norb-1)÷2
    if norb == 5
        hLFT = Matrix{Float64}(undef, norb, norb)
        for row in 1:norb
            for col in 1:norb
                hLFT[row,col] = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], row, col)
            end
        end
        perm = [4,2,1,3,5]    # change order from 0,1,-1,2,-2 to 2,1,0,-1,-2 (=x2-y2,xz,z2,yz,xy) 
        hLFT = hLFT[perm, perm]
        F0 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 2, 4)
        F2 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 3, 2)
        F4 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 4, 2)
        F = Dict(0 => F0, 2 => F2/49, 4 => F4/441)
        zeta = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "ZETA_D"], 0, 2)/219474.63  # convert from cm-1 to Hartree
    end
    if norb == 7
        hLFT = Matrix{Float64}(undef, norb, norb)
        for row in 1:norb
            for col in 1:norb
                hLFT[row,col] = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], row, col)
            end
        end
        perm = [6,4,2,1,3,5,7]    # change order from 0,1,-1,2,-2,3,-3 to 3,2,1,0,-1,-2,-3
        hLFT = hLFT[perm, perm]
        F0 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 2, 2)
        F2 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 3, 2)
        F4 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 4, 2)
        F6 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 5, 2)
        F = Dict(0 => F0, 2 => F2/15/15, 4 => F4/33/33, 6 => (5/429)^2 * F6)
        zeta = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "ZETA_F"], 0, 2)/219474.63  # convert from cm-1 to Hartree
    end
    return LFTParam(nel, norb, l, hLFT, F, zeta)
end

const HermMat = Hermitian{T, Matrix{T}} where T <: Number  # Complex hermitian (or real symmetric) matrix

function calc_H_magfield(H_fieldfree::HermMat, L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}}, B::Vector{Float64})
    H_magfield = H_fieldfree + 0.5*(L[1]*B[1] + L[2]*B[2] + L[3]*B[3]) + S[1]*B[1] + S[2]*B[2] + S[3]*B[3]
    return Hermitian(H_magfield)    # this is important for the eigensolver to yield orthogonal eigenvectors
end

function fieldfree_GS_energy(H_fieldfree::HermMat)
    energies = eigvals(H_fieldfree)
    @assert isa(energies, Vector{Float64})  # energies need to be real
    return energies[1]
end

function calc_free_energy(H_fieldfree::HermMat, L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}}, B0_mol::Vector{Float64}, T::Real)
    energies, states = calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
    kB = 3.166811563e-6    # Boltzmann constant in Eh/K
    beta = 1/(kB*T)
    energies_exp = exp.(-beta*energies)
    Z = sum(energies_exp)   # canonical partition function
    return -log(Z)/beta
end

function calc_magneticmoment_operator(L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}})
    Mel = Vector{Matrix{ComplexF64}}(undef, 3)
    for i in 1:3
        Mel[i] = -0.5*L[i] - S[i]
    end
    return Mel
end

function calc_Hderiv(L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}})
    Hderiv = Vector{Matrix{ComplexF64}}(undef, 3)
    for i in 1:3
        Hderiv[i] = 0.5*L[i] - S[i]
    end
    return Hderiv
end

"""
H_fieldfree: Hamiltonian in the absence of a magnetic field
H: Hamiltonian whose eigenstates and eigenenergies we want to calculate (relative to fieldfree ground state energy E0)
"""
function calc_solutions(H_fieldfree::HermMat, H::HermMat)
    E0 = fieldfree_GS_energy(H_fieldfree)
    solution = eigen(H)
    @assert isa(solution.values, Vector{Float64})  # energies need to be real
    energies = solution.values .- E0         # All energies relative to fieldfree GS energy
    states = solution.vectors
    return energies, states
end

calc_solutions(H_fieldfree::HermMat) = calc_solutions(H_fieldfree, H_fieldfree)

function calc_solutions_magfield(H_fieldfree::HermMat, L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}}, B0_mol::Vector{Float64})
    H_magfield = calc_H_magfield(H_fieldfree, L, S, B0_mol)
    return calc_solutions(H_fieldfree, H_magfield)
end

function calc_Zel(energies::Vector{Float64}, T::Real)
    beta = 1/(kB*T)
    energies_exp = exp.(-beta*energies)
    Z = sum(energies_exp)   # canonical partition function
end

function calc_average_magneticmoment(energies::Vector{Float64}, states::Matrix{ComplexF64}, Mel::Vector{Matrix{ComplexF64}}, T::Real)
    beta = 1/(kB*T)
    energies_exp = exp.(-beta*energies)
    Zel = sum(energies_exp)   # canonical partition function
    Mel_eigenbasis = [states'*Melcomp*states for Melcomp in Mel]
    Mel_avg = [sum(energies_exp .* diag(Mel_eigenbasis_comp))/Zel for Mel_eigenbasis_comp in Mel_eigenbasis]
    @assert norm(imag(Mel_avg)) < 1e-12    # energies need to be real
    return real(Mel_avg)
end

function calc_F_deriv1(energies::Vector{Float64}, states::Matrix{ComplexF64}, Hderiv::Vector{Matrix{ComplexF64}}, T::Real)
    beta = 1/(kB*T)
    energies_exp = exp.(-beta*energies)
    Zel = sum(energies_exp)   # canonical partition function
    Hderiv_eigenbasis = [states'*Hderiv_comp*states for Hderiv_comp in Hderiv]
    Fderiv1 = [sum(energies_exp .* diag(Hderiv_eigenbasis_comp))/Zel for Hderiv_eigenbasis_comp in Hderiv_eigenbasis]
    @assert norm(imag(Fderiv1)) < 1e-12    # need to be real
    return real(Fderiv1)
end

"""
N_polar: Number of grid points for polar angle
N_azimuthal: Number of grid points for azimuthal angle
"""
function spherical_product_grid(N_polar::Integer, N_azimuthal::Integer)
    dtheta = pi/N_polar
    dphi = 2pi/N_azimuthal
    grid = Vector{Tuple{Float64, Float64, Float64}}(undef, 0)
    for theta in (dtheta/2):dtheta:(pi-dtheta/2)
        for phi in (dphi/2):dphi:(2pi-dphi/2)
            weight = sin(theta)*dtheta*dphi
            push!(grid, (theta, phi, weight))
        end
    end
    return grid
end

"""
The function f is assumed to return a list of values (vector-valued).
f is a function of points on the unit sphere, parametrized via a polar angle theta and an azimuthal angle phi.
"""
function integrate_spherical(f::Function, grid::Vector{Tuple{Float64, Float64, Float64}})
    dim = length(f(0,0))    # number of components in the output
    integrals = zeros(dim)
    for (theta, phi, weight) in grid
        integrals += weight*f(theta, phi)
    end
    return integrals
end

function calc_dipole_matrix(R::Vector{T}) where T<:Real
    R_length = norm(R)
    idmat = Matrix(1.0I, 3, 3)
    return ((3*R*R')/(R_length^2) - idmat)/(R_length^3)
end

"""
Calculate magnetic field created by a magnetic dipole moment (everything in atomic units).
"""
function dipole_field(M::Vector{T1}, R::Vector{T2}) where {T1<:Real, T2<:Real}
    return alpha^2 * (calc_dipole_matrix(R)*M)
end

"""
B0: field strength (scalar quantity).
The B0 vector in the laboratory frame is (0,0,B0)
"""
function calc_B0_mol(B0::Real, chi::Real, theta::Real)
    Bx = -sin(theta)*cos(chi)*B0
    By = sin(theta)*sin(chi)*B0
    Bz = cos(theta)*B0
    return [Bx, By, Bz]
end

function calc_Bind_avg_lab_z(chi::Real, theta::Real, Bind_avg_mol::Vector{Float64})
    return -sin(theta)*cos(chi)*Bind_avg_mol[1] +
            sin(theta)*sin(chi)*Bind_avg_mol[2] +
            cos(theta)*         Bind_avg_mol[3]
end

"""
theta: Euler angle: Angle between z axis (molecular frame) and Z axis (lab frame)
chi: Euler angle describing rotations of the molecule around its molecular frame z axis
H_fieldfree: Hamiltonian in the absence of a magnetic field (Slater determinant basis)
L: Orbital angular momentum operators (Slater determinant basis)
S: Total spin operators (Slater determinant basis)
Mel: Electronic magnetic dipole moment operators (Slater determinant basis)
R: Vectors from the points at which we want to know the induced field (typically nuclear positions) to the paramagnetic center (atomic units = Bohr)
B0: Magnitude of the external magnetic field (atomic units)
T: Temperature (Kelvin)
"""
function calc_integrands(theta::Real, chi::Real, H_fieldfree::HermMat, L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}}, Mel::Vector{Matrix{ComplexF64}}, R::Vector{Vector{Float64}}, B0::Real, T::Real)
    B0_mol = calc_B0_mol(B0, chi, theta)
    energies, states = MagFieldLFT.calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
    Zel = calc_Zel(energies, T)
    Mel_avg_mol = calc_average_magneticmoment(energies, states, Mel, T)
    Bind_avg_mol = [dipole_field(Mel_avg_mol, R_i) for R_i in R]
    Bind_avg_lab_z = [calc_Bind_avg_lab_z(chi, theta, B_i) for B_i in Bind_avg_mol]
    return [Zel*Bind_avg_lab_z; Zel]
end

function calc_operators_SDbasis(param::LFTParam)
    l=(param.norb-1)÷2
    exc = MagFieldLFT.calc_exclists(l,param.nel)
    H_fieldfree = MagFieldLFT.calc_H_fieldfree(param, exc)
    L = MagFieldLFT.calc_L(l, exc)
    S = MagFieldLFT.calc_S(l, exc)
    Mel = MagFieldLFT.calc_magneticmoment_operator(L,S)
    return H_fieldfree, L, S, Mel
end

"""
R: Vectors from the points at which we want to know the induced field (typically nuclear positions) to the paramagnetic center (atomic units = Bohr)
B0: Magnitude of the external magnetic field (atomic units)
T: Temperature (Kelvin)
"""
function calc_Bind(param::LFTParam, R::Vector{Vector{Float64}}, B0::Real, T::Real, grid::Vector{Tuple{Float64, Float64, Float64}})
    H_fieldfree, L, S, Mel = calc_operators_SDbasis(param)
    integrands(theta, chi) = calc_integrands(theta, chi, H_fieldfree, L, S, Mel, R, B0, T)
    integrals = integrate_spherical(integrands , grid)
    numerators = integrals[1:(end-1)]
    D = integrals[end]   # denominator (normalization)
    return [N/D for N in numerators]
end

function estimate_shifts_finitefield(param::LFTParam, R::Vector{Vector{Float64}}, B0::Real, T::Real, grid::Vector{Tuple{Float64, Float64, Float64}})
    Bind_values = calc_Bind(param, R, B0, T, grid)
    return (Bind_values / B0) * 1e6    # convert to ppm
end

struct DegenerateSet
    E::Float64              # energy of the states belonging to the set
    states::Vector{Int64}   # indices of the states belonging to the set
end

function determine_degenerate_sets(energies::Vector{Float64})
    degenerate_sets = Vector{DegenerateSet}(undef, 0)
    current_energy = energies[1]
    current_states = [1]
    for i in 2:length(energies)
        if abs(energies[i]-current_energy) < 1e-10                  # threshold is arbitrarily chosen. Need to think more about this
            push!(current_states, i)
        else
            push!(degenerate_sets, DegenerateSet(current_energy, current_states))
            current_energy = energies[i]
            current_states = [i]
        end
    end
    push!(degenerate_sets, DegenerateSet(current_energy, current_states))
    return degenerate_sets
end

function calc_N_indices(number_indices::Real, M_indices::Vector{Int64})
    return [i for i in 1:number_indices if !(i in M_indices)]
end

function calc_Hderiv_part(Hderiv_eigenbasis::Vector{Matrix{ComplexF64}}, I_indices::Vector{Int64}, J_indices::Vector{Int64})
    return [Hderivcomp[I_indices, J_indices] for Hderivcomp in Hderiv_eigenbasis]
end

function calc_Hderiv_eigenbasis_denom(Hderiv_eigenbasis::Vector{Matrix{ComplexF64}}, denoms::Vector{Float64})
    Hderiv_eigenbasis_denom = deepcopy(Hderiv_eigenbasis)
    for n in 1:length(denoms)
        for i in 1:3
            Hderiv_eigenbasis_denom[i][:, n] /= denoms[n]
        end
    end
    return Hderiv_eigenbasis_denom
end

function calc_F_deriv2(energies::Vector{Float64}, states::Matrix{ComplexF64}, Hderiv::Vector{Matrix{ComplexF64}}, T::Real)
    beta = 1/(kB*T)
    Zel = calc_Zel(energies, T)
    Hderiv_eigenbasis = [states'*Hderivcomp*states for Hderivcomp in Hderiv]
    degenerate_sets = determine_degenerate_sets(energies)
    Fderiv2 = im*zeros(3, 3)
    for M in degenerate_sets
        M_indices = M.states
        N_indices = calc_N_indices(length(energies), M_indices)
        Boltzmann_factor = exp(-beta*M.E)
        Hderiv_MM = calc_Hderiv_part(Hderiv_eigenbasis, M_indices, M_indices)
        Hderiv_MN = calc_Hderiv_part(Hderiv_eigenbasis, M_indices, N_indices)
        Hderiv_eigenbasis_denom = calc_Hderiv_eigenbasis_denom(Hderiv_eigenbasis, energies .- M.E)
        Hderiv_MN_denom = calc_Hderiv_part(Hderiv_eigenbasis_denom, M_indices, N_indices)
        for k in 1:3
            for l in 1:3
                part1 = 0.5*beta*tr(Hderiv_MM[k] * Hderiv_MM[l]')
                part2 = tr(Hderiv_MN[k] * Hderiv_MN_denom[l]')
                Fderiv2[k, l] += Boltzmann_factor * (part1 + part2)
            end
        end
    end
    Fderiv2 *= -1/Zel
    Fderiv1 = calc_F_deriv1(energies, states, Hderiv, T)
    Fderiv2 += 0.5*beta* Fderiv1*Fderiv1'
    Fderiv2 += transpose(Fderiv2)  # symmetrization
    @assert norm(imag(Fderiv2)) < 1e-5
    return real(Fderiv2)
end

function calc_F_deriv3(energies::Vector{Float64}, states::Matrix{ComplexF64}, Hderiv::Vector{Matrix{ComplexF64}}, T::Real)
    beta = 1/(kB*T)
    Zel = calc_Zel(energies, T)
    Hderiv_eigenbasis = [states'*Hderivcomp*states for Hderivcomp in Hderiv]
    degenerate_sets = determine_degenerate_sets(energies)
    Fderiv3 = im*zeros(3, 3, 3)
    for M in degenerate_sets
        M_indices = M.states
        N_indices = [i for i in 1:length(energies) if !(i in M_indices)]
        Boltzmann_factor = exp(-beta*M.E)
        Hderiv_MM = calc_Hderiv_part(Hderiv_eigenbasis, M_indices, M_indices)
        Hderiv_MN = calc_Hderiv_part(Hderiv_eigenbasis, M_indices, N_indices)
        Hderiv_NN = calc_Hderiv_part(Hderiv_eigenbasis, N_indices, N_indices)
        Hderiv_eigenbasis_denom = calc_Hderiv_eigenbasis_denom(Hderiv_eigenbasis, energies .- M.E)
        Hderiv_MN_denom = calc_Hderiv_part(Hderiv_eigenbasis_denom, M_indices, N_indices)
        for l in 1:3
            for k in 1:3
                for j in 1:3
                    part1 = (1/6)*beta^2*tr(Hderiv_MM[l] * Hderiv_MM[k] * Hderiv_MM[j])
                    part2 = beta*tr(Hderiv_MN_denom[l] * Hderiv_MN[k]' * Hderiv_MM[j])
                    part3 = - tr(Hderiv_MN_denom[l] * Hderiv_MN_denom[k]' * Hderiv_MM[j])
                    part4 = tr(Hderiv_MN_denom[l] * Hderiv_NN[k] * Hderiv_MN_denom[j]')
                    Fderiv3[l,k,j] += Boltzmann_factor * (part1 + part2 + part3 + part4)
                end
            end
        end
    end
    Fderiv3 *= 1/Zel
    Fderiv1 = calc_F_deriv1(energies, states, Hderiv, T)
    Fderiv2 = calc_F_deriv2(energies, states, Hderiv, T)
    for l in 1:3
        for k in 1:3
            for j in 1:3
                Fderiv3[l,k,j] += -(1/6)*beta^2*Fderiv1[l]*Fderiv1[k]*Fderiv1[j]
                Fderiv3[l,k,j] += 0.5*beta*Fderiv1[l]*Fderiv2[k,j]
            end
        end
    end
    # symmetrization:
    nindices = 3
    Fderiv3_symmetrized = im*zeros(3,3,3)
    for k in 1:factorial(nindices)   # loop over all permutations of three indices
        Fderiv3_symmetrized += permutedims(Fderiv3, Permutation(nindices,k))
    end
    @assert norm(imag(Fderiv3_symmetrized)) < 1e-4
    return real(Fderiv3_symmetrized)
end

function calc_F_deriv4(energies::Vector{Float64}, states::Matrix{ComplexF64}, Hderiv::Vector{Matrix{ComplexF64}}, T::Real)
    beta = 1/(kB*T)
    Zel = calc_Zel(energies, T)
    Hderiv_eigenbasis = [states'*Hderivcomp*states for Hderivcomp in Hderiv]
    degenerate_sets = determine_degenerate_sets(energies)
    Fderiv4 = im*zeros(3, 3, 3, 3)
    for M in degenerate_sets
        M_indices = M.states
        N_indices = [i for i in 1:length(energies) if !(i in M_indices)]
        Boltzmann_factor = exp(-beta*M.E)
        Hderiv_MM = calc_Hderiv_part(Hderiv_eigenbasis, M_indices, M_indices)
        Hderiv_MN = calc_Hderiv_part(Hderiv_eigenbasis, M_indices, N_indices)
        Hderiv_NN = calc_Hderiv_part(Hderiv_eigenbasis, N_indices, N_indices)
        Hderiv_eigenbasis_denom = calc_Hderiv_eigenbasis_denom(Hderiv_eigenbasis, energies .- M.E)
        Hderiv_eigenbasis_denom2 = calc_Hderiv_eigenbasis_denom(Hderiv_eigenbasis, (energies .- M.E) .* (energies .- M.E))
        Hderiv_MN_denom = calc_Hderiv_part(Hderiv_eigenbasis_denom, M_indices, N_indices)
        Hderiv_MN_denom2 = calc_Hderiv_part(Hderiv_eigenbasis_denom2, M_indices, N_indices)
        Hderiv_NN_denom = calc_Hderiv_part(Hderiv_eigenbasis_denom, N_indices, N_indices)
        for l in 1:3, k in 1:3, j in 1:3, i in 1:3
            part1 = (1/24)*beta^3*tr(Hderiv_MM[l] * Hderiv_MM[k] * Hderiv_MM[j] * Hderiv_MM[i])
            part2 = (1/2)*beta^2*tr(Hderiv_MN_denom[l] * Hderiv_MN[k]' * Hderiv_MM[j] * Hderiv_MM[i])
            part3 = -beta*tr(Hderiv_MN_denom2[l] * Hderiv_MN[k]' * Hderiv_MM[j] * Hderiv_MM[i])
            part4 = tr(Hderiv_MN_denom2[l] * Hderiv_MN_denom[k]' * Hderiv_MM[j] * Hderiv_MM[i])
            part5 = 0.5*beta*tr(Hderiv_MN_denom[l] * Hderiv_MN[k]' * Hderiv_MN_denom[j] * Hderiv_MN[i]')
            part6 = beta*tr(Hderiv_MN_denom[l] * Hderiv_NN[k] * Hderiv_MN_denom[j]' * Hderiv_MM[i])
            part7 = -tr(Hderiv_MN_denom2[l] * Hderiv_MN[k]' * Hderiv_MN_denom[j] * Hderiv_MN[i]')
            part8 = -tr(Hderiv_MN_denom[l] * Hderiv_NN[k] * Hderiv_MN_denom2[j]' * Hderiv_MM[i])
            part9 = part8'
            part10 = tr(Hderiv_MN_denom[l] * Hderiv_NN_denom[k] * Hderiv_NN_denom[j] * Hderiv_MN[i]')
            Fderiv4[l,k,j,i] += Boltzmann_factor * (part1 + part2 + part3 + part4 + part5 +
                                                    part6 + part7 + part8 + part9 + part10)
        end
    end
    Fderiv4 *= -1/Zel
    Fderiv1 = calc_F_deriv1(energies, states, Hderiv, T)
    Fderiv2 = calc_F_deriv2(energies, states, Hderiv, T)
    Fderiv3 = calc_F_deriv3(energies, states, Hderiv, T)
    for l in 1:3, k in 1:3, j in 1:3, i in 1:3
        Fderiv4[l,k,j,i] += (1/24)*beta^3*Fderiv1[l]*Fderiv1[k]*Fderiv1[j]*Fderiv1[i] -
                        (1/4)*beta^2*Fderiv1[l]*Fderiv1[k]*Fderiv2[j,i] +
                        (1/6)*beta*Fderiv1[l]*Fderiv3[k,j,i] +
                        (1/8)*beta*Fderiv2[l,k]*Fderiv2[j,i]
    end
    # symmetrization:
    nindices = 4
    Fderiv4_symmetrized = im*zeros(3,3,3,3)
    for k in 1:factorial(nindices)   # loop over all permutations of three indices
        Fderiv4_symmetrized += permutedims(Fderiv4, Permutation(nindices,k))
    end
    @assert norm(imag(Fderiv4_symmetrized))/norm(real(Fderiv4_symmetrized)) < 1e-10
    return real(Fderiv4_symmetrized)
end

function F_deriv_param2states(calc_F_derivx::Function)
    function calc_F_deriv_param(param::LFTParam, T::Real, B0_mol::Vector{Float64})
        H_fieldfree, L, S, Mel = calc_operators_SDbasis(param)
        Hderiv = -Mel
        energies, states = calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
        return calc_F_derivx(energies, states, Hderiv, T)
    end
    return calc_F_deriv_param
end

calc_F_deriv1(param::LFTParam, T::Real, B0_mol::Vector{Float64}) = F_deriv_param2states(calc_F_deriv1)(param, T, B0_mol)
calc_F_deriv2(param::LFTParam, T::Real, B0_mol::Vector{Float64}) = F_deriv_param2states(calc_F_deriv2)(param, T, B0_mol)
calc_F_deriv3(param::LFTParam, T::Real, B0_mol::Vector{Float64}) = F_deriv_param2states(calc_F_deriv3)(param, T, B0_mol)
calc_F_deriv4(param::LFTParam, T::Real, B0_mol::Vector{Float64}) = F_deriv_param2states(calc_F_deriv4)(param, T, B0_mol)

function calc_susceptibility_vanVleck(param::LFTParam, T::Real)
    B = [0.0, 0.0, 0.0]
    Fderiv2 = calc_F_deriv2(param, T, B)
    return -4pi*alpha^2 * Fderiv2
end

"""
Returns the chemical shifts in ppm calculated according to the Kurland-McGarvey equation (point-dipole approximation)
R: Vectors from the points at which we want to know the induced field (typically nuclear positions) to the paramagnetic center (atomic units = Bohr)
T: Temperature (Kelvin)
"""
function calc_shifts_KurlandMcGarvey(param::LFTParam, R::Vector{Vector{Float64}}, T::Real)
    chi = calc_susceptibility_vanVleck(param, T)
    shifts = Vector{Float64}(undef, 0)
    for Ri in R
        D = calc_dipole_matrix(Ri)
        sigma = -(1/(4pi)) * chi * D
        shift = -(1/3)*tr(sigma)
        push!(shifts, shift)
    end
    shifts *= 1e6    # convert to ppm
    return shifts
end

function Tesla2au(B::Real)
    return B/2.35051756758e5
end

"""
nu: Resonance frequency in MHz
"""
function MHz2Tesla(nu::Real)
    gamma = 42.577478518     # gyromagnetic ratio of the proton in MHz/T
    return nu/gamma
end

"""
nu: Resonance frequency in MHz
"""
function MHz2au(nu::Real)
    return Tesla2au(MHz2Tesla(nu))
end

"""
C: Coefficients of the state in the Slater determinant basis
U: Matrix whose columns are coefficients of basis states in the Slater determinant basis.
   This matrix must be unitary, i.e., the basis must be orthonormal!
labels: Unique identifiers for the basis states (e.g. quantum numbers)
thresh: Total percentage to which the printed contributions must at least sum up to (default: 0.98)
"""
function print_composition(C::Vector{T1}, U::Matrix{T2}, labels::Vector{String}, thresh::Number=0.98, io::IO=stdout) where {T1 <: Number, T2 <: Number}
    U_list = [U[:,i:i] for i in 1:size(U)[2]]
    print_composition(C, U_list, labels, thresh, io)
end

"""
Same as before, but now you don't pass a square matrix U with basis vectors, but a list of (in general rectangular)
matrices U. This is used if some labels are used for whole groups of basis vectors (e.g., if
we want to print the percentage of a certain total spin).
"""
function print_composition(C::Vector{T1}, U_list::Vector{Matrix{T2}}, labels::Vector{String}, thresh::Number=0.99, io::IO=stdout) where {T1 <: Number, T2 <: Number}
    percentages = [real(C'*U*U'*C) for U in U_list]
    p = sortperm(percentages, rev=true)
    percentages_sorted = percentages[p]
    labels_sorted = labels[p]
    total_percentage = 0.0
    i = 1
    while total_percentage < thresh
        @printf(io, "%6.2f%%  %s\n", percentages_sorted[i]*100, labels_sorted[i])
        total_percentage += percentages_sorted[i]
        i += 1
    end
end

"""
This function assumes that the eigenvalues are sorted (i.e., equal eigenvalues have neighboring indices)
"""
function group_eigenvalues(values::Vector{T}, thresh::Real=1.0e-8) where T<:Real
    indices = Vector{Vector{Int64}}(undef, 0)
    unique_values = Vector{T}(undef, 0)
    current_value = values[1]
    current_indices = [1]
    for i in 2:length(values)
        if abs(values[i] - current_value) < thresh
            push!(current_indices, i)
        else
            push!(indices, current_indices)
            push!(unique_values, current_value)
            current_indices = [i]
            current_value = values[i]
        end
    end
    push!(indices, current_indices)
    push!(unique_values, current_value)
    return unique_values, indices
end

function adapt_basis(C_list::Vector{Matrix{T1}}, labels_list::Vector{Vector{Float64}}, op::HermMat) where {T1<:Number}
    C_list_new = Vector{Matrix{ComplexF64}}(undef, 0)
    labels_list_new = Vector{Vector{Float64}}(undef, 0)
    for i in 1:length(C_list)
        C = C_list[i]
        op_C = Hermitian(C'*op*C)
        vals, vecs = eigen(op_C)
        unique_vals, indices = group_eigenvalues(vals)
        for j in 1:length(unique_vals)
            V = vecs[:, indices[j]]
            labels = deepcopy(labels_list[i])
            push!(labels, unique_vals[j])
            push!(C_list_new, C*V)
            push!(labels_list_new, labels)
        end
    end
    return C_list_new, labels_list_new
end

"""
Turn a value that might almost be integer or half-integer into one that is exactly integer or half-integer.
Type of returned value: Float64
"""
function exactspinQN(value::Real)
    return round(2*value+1e-10)/2     # add small number before rounding to always get 0.0 instead of -0.0
end

"""
Given an eigenvalue of a squared angular momentum operator J2, i.e. value = J(J+1),
return the corresponding total angular momentum quantum number J.
"""
function J2_to_J(value::Real)
    J = sqrt(value+0.25)-0.5
    return exactspinQN(J)
end

function prettylabels_Hnonrel_S2_Sz(labels_list::Vector{Vector{Float64}})
    return [@sprintf("E = %10.6f, S = %4.1f, M_S = %4.1f", l[1], J2_to_J(l[2]), exactspinQN(l[3])) for l in labels_list]
end

function adapt_basis_Hnonrel_S2_Sz(param::LFTParam)
    exc = calc_exclists(param.l,param.nel)
    Hnonrel = MagFieldLFT.calc_H_nonrel(param, exc)
    Sx, Sy, Sz = calc_S(param.l, exc)
    S2 = Sx*Sx + Sy*Sy + Sz*Sz
    C_list1, labels_list1 = adapt_basis([Matrix{Float64}(1.0I, exc.Dim, exc.Dim)], [Vector{Float64}(undef, 0)], Hermitian(Hnonrel))
    C_list2, labels_list2 = adapt_basis(C_list1, labels_list1, Hermitian(S2))
    C_list3, labels_list3 = adapt_basis(C_list2, labels_list2, Hermitian(Sz))
    str_labels_list3 = prettylabels_Hnonrel_S2_Sz(labels_list3)
    return C_list3, str_labels_list3
end

term_sym = Dict(0 => "S",
                1 => "P",
                2 => "D",
                3 => "F",
                4 => "G",
                5 => "H",
                6 => "I",
                7 => "K",
                8 => "L",
                9 => "M",
                10 => "N",
                11 => "O",
                12 => "Q",
                13 => "R",
                14 => "T",
                15 => "U",
                16 => "V")

function spinQNlabel(S::Real)
    if abs(S-round(S)) < 1e-10
        label = "$(Int64(round(S)))"
    elseif abs(2S-round(2S)) < 1e-10
        numerator = Int64(round(2S))
        label = "$(numerator)/2"
    else
        println(S)
        error("Spin quantum number may only take integer or half-integer values!")
    end
    return label
end

"""
format: How are the states labeled. Default: "QN" (quantum numbers). Other option: "term_symbols".
"""
function prettylabels_L2_S2_J2_Jz(labels_list::Vector{Vector{Float64}}, format::String="QN")
    if format == "QN"
        return [@sprintf("(L, S, J, M_J) = (%4.1f,%4.1f,%4.1f,%4.1f)", J2_to_J(l[1]), J2_to_J(l[2]), J2_to_J(l[3]), exactspinQN(l[4])) for l in labels_list]
    elseif format == "term_symbols"
        return [@sprintf("Term: %d%s%s, M_J = %s", Int(round(2*J2_to_J(l[2])+1)), term_sym[Int(round(J2_to_J(l[1])))], spinQNlabel(J2_to_J(l[3])), spinQNlabel(l[4])) for l in labels_list]
    end
end

function adapt_basis_L2_S2_J2_Jz(param::LFTParam, format::String="QN")
    exc = calc_exclists(param.l,param.nel)
    Sx, Sy, Sz = calc_S(param.l, exc)
    Lx, Ly, Lz = calc_L(param.l, exc)
    Jx = Lx+Sx
    Jy = Ly+Sy
    Jz = Lz+Sz
    S2 = Sx*Sx + Sy*Sy + Sz*Sz
    L2 = Lx*Lx + Ly*Ly + Lz*Lz
    J2 = Jx*Jx + Jy*Jy + Jz*Jz
    C_list1, labels_list1 = adapt_basis([Matrix{Float64}(1.0I, exc.Dim, exc.Dim)], [Vector{Float64}(undef, 0)], Hermitian(L2))
    C_list2, labels_list2 = adapt_basis(C_list1, labels_list1, Hermitian(S2))
    C_list3, labels_list3 = adapt_basis(C_list2, labels_list2, Hermitian(J2))
    C_list4, labels_list4 = adapt_basis(C_list3, labels_list3, Hermitian(Jz))
    str_labels_list4 = prettylabels_L2_S2_J2_Jz(labels_list4, format)
    return C_list4, str_labels_list4
end

function Brillouin(S::Float64, T::Float64, B0::Float64)

    muB = 0.5
    ge = 2.0

    k_B = 3/2*kB*T /(ge*muB*S*(S+1)*B0)
    a = muB*ge*B0/(2*kB*T)
    if B0 != 0
        Br = k_B*((2*S+1)/tanh((2*S+1)*a) - 1/tanh(a))
    else
        Br = 1.0
    end

    return Br
end

function Brillouin_truncated(S::Float64, T::Float64, B0::Float64)
    Br = 1 + B0^2/(240*(kB*T)^2*S*(S+1)) * (1-(2*S+1)^4)
    return Br
end

function orientation_tensor(B0::Float64, T::Float64, chi::Array{Float64, 2})
    #field-induced self-orientation tensor 
    #(eq 112 from Parigi, G. et al, Progress in Nuclear Magnetic Resonance Spectroscopy 114–115 (2019) 211–236) 

    w,v = eigen(chi)
    a = (B0^2)/(5*mu0*kB*T)
    chiiso = (1/3)*tr(Diagonal(w))
    Pw = zeros(3,3)
    for i in 1:3
    	Pw[i,i]= (1 + a*(w[i]-chiiso))
    end
    P = v * Pw * v'
    return P
end

function trace_ord2(tensor::Array{Float64, 4})
    # computes the order 2 trace of a supersymmetric fourth order tensor

    trace = @tensor begin
        trace = tensor[i, i, j, j]
    end

    return trace
end

function product_ord3(tensor::Array{Float64, 4}, Dip::Array{Float64, 2})
    #dot product between two fourth order tensors

    sigma = zeros(Float64, 3, 3, 3, 3)

    @tensor begin
        sigma[l, m, n, k] := tensor[l, m, n, q] * Dip[q, k]
    end

    return sigma
end

function calc_shifts_KurlandMcGarvey_ord4(param::LFTParam, R::Vector{Vector{Float64}}, T::Real, B0::Real, direct::Bool=false, selforient::Bool=false)
    #pcs calculation with Kurland-McGarvey equation 
    #the saturation effect is accounted for with fourth order tensor (determined via analytical equation)

    beta = 1/(kB*T)

    chi = calc_susceptibility_vanVleck(param, T)
    if direct
        chi3 = -4pi*alpha^2*calc_F_deriv4(param, T, [0.0,0.0,0.0]) 
    end
  
    shifts = Vector{Float64}(undef, 0)
    for Ri in R
        Dip = calc_dipole_matrix(Ri)
        sigma = -(1/(4pi)) * chi * Dip
        shift = -(1/3)*tr(sigma)

        if direct
            tau = -(1/(4pi)) * (1/6) * product_ord3(chi3,Dip)
            shift += - (1/5)*trace_ord2(tau)*B0^2
        end

        if selforient
            shift += (1/45 * beta/mu0 *tr(sigma)*tr(chi) - 1/15 * beta/mu0 * tr(sigma*chi))*B0^2
        end

        push!(shifts, shift)
    end
    shifts *= 1e6    # convert to ppm
    return shifts
end

function calc_shifts_KurlandMcGarvey_Br(param::LFTParam, R::Vector{Vector{Float64}}, T::Real, B0::Float64, S::Float64, direct::Bool=false, selforient::Bool=false)
    #pcs calculation with Kurland-McGarvey equation
    #the saturation effect is accounted for with Brillouin equation

    beta = 1/(kB*T)

    #Br = Brillouin(S, T, B0)
    Br = Brillouin_truncated(S, T, B0)
    chi = calc_susceptibility_vanVleck(param, T)

    shifts = Vector{Float64}(undef, 0)
    for Ri in R
        Dip = calc_dipole_matrix(Ri)
        sigma = -(1/(4pi)) * chi * Dip
        shift = -(1/3)*tr(sigma)

        if direct
            shift = -(1/3)*tr(sigma .* Br)
        end

        if selforient
            shift += (1/45 * beta/mu0 *tr(sigma)*tr(chi) - 1/15 * beta/mu0 * tr(sigma*chi))*B0^2 
        end

        push!(shifts, shift)
    end
    shifts *= 1e6    # convert to ppm
    return shifts
end

function calc_dyadics(s::Float64, D::Matrix{Float64}, T::Real, quadruple::Bool)

    Sp = calc_splusminus(s, +1)
    Sm = calc_splusminus(s, -1)
    Sz = calc_sz(s)

    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5im * (Sp-Sm)

    Hderiv = [Sx, Sy, Sz]

    S = cat(Sx, Sy, Sz; dims=3)
    StDS = sum(D[i, j] * S[:, :, i] * S[:, :, j] for i in 1:3, j in 1:3)

    solution = eigen(StDS)
    energies = solution.values
    states = solution.vectors

    SS = -calc_F_deriv2(energies, states, Hderiv, T)

    if quadruple

        SSSS = -calc_F_deriv4(energies, states, Hderiv, T)

        return SS, SSSS
    else
        return SS
    end

end


function calc_contactshift_fielddep_Br(s::Float64, Aiso::Matrix{Float64}, g::Matrix{Float64}, D::Matrix{Float64}, T::Real, B0::Float64, direct::Bool=false, selforient::Bool=false)

    gammaI = 2.6752e8*1e-6 
    gammaI *= 2.35051756758e5

    beta = 1/(kB*T)

    SS = calc_dyadics(s, D, T, false)

    #Br = Brillouin(s, T, B0)
    Br = Brillouin_truncated(s, T, B0)
    sigma = zeros(length(Aiso), 3, 3)

    chi = pi*MagFieldLFT.alpha^2 * g * SS * g'

    shiftcon = Float64[]

    for (i, Aiso_val) in enumerate(Aiso)

        for l in 1:3, k in 1:3, o in 1:3, p in 1:3
            if k == p
                sigma[i, l, k] += -(1/2) * g[l, o] * (Aiso_val*2pi) *(1/gammaI) * SS[o, p]
            end
        end

        con = 0

        if selforient
            con = ((1/45 * beta/mu0 *tr(sigma[i,:,:])*tr(chi) - 1/15 * beta/mu0 * tr(sigma[i,:,:]*chi))*B0^2) * 1e6
        end

        if direct
            sigma[i,:,:] *= Br
        end

        con += -1/3 * tr(sigma[i, :, :]) * 1e6
        
        push!(shiftcon, con)
    end

    return shiftcon
end




function calc_contactshift_fielddep(s::Float64, Aiso::Matrix{Float64}, g::Matrix{Float64}, D::Matrix{Float64}, T::Real, B0::Float64, direct::Bool=false, selforient::Bool=false)

    gammaI = 2.6752e8*1e-6 
    gammaI *= 2.35051756758e5

    beta = 1/(kB*T)

    SS, SSSS = calc_dyadics(s, D, T, true)
    
    chi = pi*MagFieldLFT.alpha^2 * g * SS * g'

    sigma1 = zeros(length(Aiso), 3, 3)
    sigma3 = zeros(length(Aiso), 3, 3, 3, 3)

    shiftcon = Float64[]

    for (i, Aiso_val) in enumerate(Aiso)

        for l in 1:3, m in 1:3, n in 1:3, k in 1:3, o in 1:3, p in 1:3, q in 1:3, r in 1:3
            if k == p && m == 1 && n == 1 && q == 1 && r == 1
                sigma1[i, l, k] += -(1/2) * g[l, o] * (Aiso_val*2pi) * (1/gammaI) * SS[o, p]
            end

            if k == r
                sigma3[i, l, m, n, k] += -(1/8) * g[l, o] * g[m, p] * g[n, q] * (Aiso_val*2pi) *(1/gammaI) * SSSS[o, p, q, r]
            end
        end

        con = -1/3 * tr(sigma1[i, :, :]) * 1e6

        if direct
            con += - 1/30 * trace_ord2(sigma3[i, :, :, :, :]) * B0^2 * 1e6
        end

        if selforient
            con += ((1/45 * beta/mu0 *tr(sigma1[i,:,:])*tr(chi) - 1/15 * beta/mu0 * tr(sigma1[i,:,:]*chi))*B0^2) * 1e6
        end

        push!(shiftcon, con)
    end

    return shiftcon
end


end