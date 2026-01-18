import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ============================================================
# KONSTANTA & LATTICE HONEYCOMB
# ============================================================

hbar = 1.054571817e-34
eV_to_J = 1.602176634e-19

def build_lattice(a=1.0):
    a1 = jnp.array([jnp.sqrt(3.0) * a / 2.0,  3.0 * a / 2.0])
    a2 = jnp.array([-jnp.sqrt(3.0) * a / 2.0, 3.0 * a / 2.0])

    A = jnp.stack([a1, a2], axis=0)
    B = 2.0 * jnp.pi * jnp.linalg.inv(A).T
    b1, b2 = B[0], B[1]

    deltas = jnp.array([
        [0.0,        -a],
        [ jnp.sqrt(3.0) * a / 2.0,  a / 2.0],
        [-jnp.sqrt(3.0) * a / 2.0,  a / 2.0],
    ])

    return a, a1, a2, b1, b2, deltas

# pakai a fisik ~1.5 Å
a, a1, a2, b1, b2, deltas = build_lattice(a=1.5e-10)

# ============================================================
# GRAPHENE: HAMILTONIAN & NEWTON REFINEMENT K
# ============================================================

def f_k(k, tJ, deltas):
    phase = deltas @ k
    return -tJ * jnp.sum(jnp.exp(1j * phase))

def h_graphene(k, tJ, deltas, eps_diag):
    f = f_k(k, tJ, deltas)
    return jnp.array(
        [[eps_diag, f],
         [jnp.conj(f), eps_diag]],
        dtype=jnp.complex128,
    )

def eigsys_graphene(k, tJ, deltas, eps_diag):
    vals, vecs = jnp.linalg.eigh(h_graphene(k, tJ, deltas, eps_diag))
    return vals, vecs

def newton_refine_dirac(k0, tJ, deltas, eps_diag, max_iter=20, tol=1e-14):
    def body_fun(state):
        k, i = state
        fk = f_k(k, tJ, deltas)

        def f_real_imag(k_):
            fk_ = f_k(k_, tJ, deltas)
            return jnp.array([jnp.real(fk_), jnp.imag(fk_)])

        J = jax.jacobian(f_real_imag)(k)
        rhs = -jnp.array([jnp.real(fk), jnp.imag(fk)])
        delta = jnp.linalg.solve(J, rhs)
        k_new = k + delta
        return (k_new, i + 1)

    def cond_fun(state):
        k, i = state
        fk = f_k(k, tJ, deltas)
        return jnp.logical_and(jnp.linalg.norm(fk) > tol, i < max_iter)

    k_ref, iters = jax.lax.while_loop(cond_fun, body_fun, (k0, 0))
    return k_ref, iters

# ============================================================
# GRAPHENE: vF RING FIT
# ============================================================

def ring_k_points(K, q_abs, directions):
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, directions, endpoint=False)
    qx = q_abs * jnp.cos(angles)
    qy = q_abs * jnp.sin(angles)
    q = jnp.stack([qx, qy], axis=-1)
    return K + q

def graphene_band_energy(k, tJ, deltas, eps_diag, band_index):
    vals, _ = eigsys_graphene(k, tJ, deltas, eps_diag)
    return vals[band_index]

def fit_vF_ring(K, tJ, deltas, eps_diag, a, q_abs_rel=5e-3, directions=512):
    q_abs = q_abs_rel * (2.0 * jnp.pi / a)
    k_ring = ring_k_points(K, q_abs, directions)

    band_index = 1  # conduction
    E_ring = jax.vmap(lambda k: graphene_band_energy(k, tJ, deltas, eps_diag, band_index))(k_ring)

    E0 = graphene_band_energy(K, tJ, deltas, eps_diag, band_index)
    dE = E_ring - E0

    vF_fit = jnp.mean(dE / (hbar * q_abs))
    spread = jnp.std(dE / (hbar * q_abs))
    return vF_fit, spread

# ============================================================
# GRAPHENE: BERRY PHASE & LOCAL CURVATURE
# ============================================================

def berry_phase_ring(K, eigsys, a, q_abs_rel=5e-3, directions=512, band_index=0):
    q_abs = q_abs_rel * (2.0 * jnp.pi / a)
    k_ring = ring_k_points(K, q_abs, directions)

    def get_vec(k):
        vals, vecs = eigsys(k)
        v = vecs[:, band_index]
        return v / jnp.linalg.norm(v)

    vecs = jax.vmap(get_vec)(k_ring)
    inner = jnp.sum(jnp.conj(vecs) * jnp.roll(vecs, -1, axis=0), axis=-1)
    U = inner / jnp.abs(inner)
    gamma = jnp.angle(jnp.prod(U))
    return q_abs, gamma

def berry_curvature_two_radii(K, eigsys, a, base_q_rel=5e-3, directions=512, band_index=0):
    r1 = base_q_rel
    r2 = 2.0 * base_q_rel

    _, g1 = berry_phase_ring(K, eigsys, a, q_abs_rel=r1, directions=directions, band_index=band_index)
    _, g2 = berry_phase_ring(K, eigsys, a, q_abs_rel=r2, directions=directions, band_index=band_index)

    A1 = jnp.pi * (r1 / a)**2
    A2 = jnp.pi * (r2 / a)**2
    F_est = (g2 - g1) / (A2 - A1)
    return F_est

# ============================================================
# BZ MAP (FUKUI) & CHERN (HONEYCOMB)
# ============================================================

def berry_curvature_bz(eigsys, b1, b2, Nk=41, band_index=0):
    n = jnp.arange(Nk)
    k1, k2 = jnp.meshgrid(n, n, indexing="ij")
    kvecs = (k1[..., None] / Nk) * b1 + (k2[..., None] / Nk) * b2
    k_flat = kvecs.reshape(-1, 2)

    def get_vec(k):
        vals, vecs = eigsys(k)
        v = vecs[:, band_index]
        return v / jnp.linalg.norm(v)

    vecs_flat = jax.vmap(get_vec)(k_flat)
    vecs = vecs_flat.reshape(Nk, Nk, 2)

    inner1 = jnp.sum(jnp.conj(vecs) * jnp.roll(vecs, -1, axis=0), axis=-1)
    inner2 = jnp.sum(jnp.conj(vecs) * jnp.roll(vecs, -1, axis=1), axis=-1)

    U1 = inner1 / jnp.abs(inner1)
    U2 = inner2 / jnp.abs(inner2)

    F = jnp.angle(
        U1
        * jnp.roll(U2, -1, axis=0)
        * jnp.conj(jnp.roll(U1, -1, axis=1))
        * jnp.conj(U2)
    )
    return F

# ============================================================
# GRAPHENE: MASSIVE DIRAC HAMILTONIAN (VALLEY K)
# ============================================================

def make_hamiltonian_mass(f_k_fun, eps_diag, mJ):
    def h_k_m(k):
        f = f_k_fun(k)
        return jnp.array(
            [[eps_diag + mJ, f],
             [jnp.conj(f),   eps_diag - mJ]],
            dtype=jnp.complex128,
        )

    def eigsys_m(k):
        vals, vecs = jnp.linalg.eigh(h_k_m(k))
        return vals, vecs

    def energies_m(k):
        vals, _ = eigsys_m(k)
        return vals

    return h_k_m, energies_m, eigsys_m

# ============================================================
# QWZ MODEL (CHERN INSULATOR)
# ============================================================

def qwz_hamiltonian(m):
    def h_k(k):
        kx, ky = k
        dx = jnp.sin(kx)
        dy = jnp.sin(ky)
        dz = m + jnp.cos(kx) + jnp.cos(ky)
        return jnp.array(
            [[dz, dx - 1j * dy],
             [dx + 1j * dy, -dz]],
            dtype=jnp.complex128,
        )

    def eigsys(k):
        vals, vecs = jnp.linalg.eigh(h_k(k))
        return vals, vecs

    return h_k, eigsys

def berry_curvature_bz_square(eigsys, Nk=41, band_index=0):
    k_lin = jnp.linspace(-jnp.pi, jnp.pi, Nk, endpoint=False)
    kx, ky = jnp.meshgrid(k_lin, k_lin, indexing="ij")
    kvecs = jnp.stack([kx, ky], axis=-1)
    k_flat = kvecs.reshape(-1, 2)

    def get_vec(k):
        vals, vecs = eigsys(k)
        v = vecs[:, band_index]
        return v / jnp.linalg.norm(v)

    vecs_flat = jax.vmap(get_vec)(k_flat)
    vecs = vecs_flat.reshape(Nk, Nk, 2)

    inner1 = jnp.sum(jnp.conj(vecs) * jnp.roll(vecs, -1, axis=0), axis=-1)
    inner2 = jnp.sum(jnp.conj(vecs) * jnp.roll(vecs, -1, axis=1), axis=-1)

    U1 = inner1 / jnp.abs(inner1)
    U2 = inner2 / jnp.abs(inner2)

    F = jnp.angle(
        U1
        * jnp.roll(U2, -1, axis=0)
        * jnp.conj(jnp.roll(U1, -1, axis=1))
        * jnp.conj(U2)
    )
    return F

def qwz_bz_map_and_chern(m, Nk=61):
    h_k_qwz, eigsys_qwz = qwz_hamiltonian(m)
    F_bz_qwz = berry_curvature_bz_square(eigsys_qwz, Nk=Nk, band_index=0)
    chern_qwz = jnp.sum(F_bz_qwz) / (2.0 * jnp.pi)
    return F_bz_qwz, chern_qwz

# ============================================================
# HALDANE MODEL (ORIENTED NNN, NORMALIZED CHERN)
# ============================================================

def make_haldane_hamiltonian_oriented(a1, a2, deltas, tJ, t2J, phi, M_J):
    c1 = a1
    c2 = a2
    c3 = a2 - a1
    oriented_nnn = jnp.stack([c1, c2, c3], axis=0)

    def h_k(k):
        phase_nn = deltas @ k
        f = -tJ * jnp.sum(jnp.exp(1j * phase_nn))

        phase_nnn = oriented_nnn @ k
        sum_cos = jnp.sum(jnp.cos(phase_nnn))
        sum_sin = jnp.sum(jnp.sin(phase_nnn))

        d0 = t2J * jnp.cos(phi) * sum_cos
        dz = M_J - t2J * jnp.sin(phi) * sum_sin

        return jnp.array(
            [[d0 + dz, f],
             [jnp.conj(f), d0 - dz]],
            dtype=jnp.complex128,
        )

    def eigsys(k):
        vals, vecs = jnp.linalg.eigh(h_k(k))
        return vals, vecs

    return h_k, eigsys

def haldane_bz_map_and_chern_oriented(
    a1,
    a2,
    deltas,
    b1,
    b2,
    t_eV=1.0,
    t2_eV=0.1,
    phi=jnp.pi / 2,
    M_eV=0.0,
    Nk=61,
):
    tJ = t_eV * eV_to_J
    t2J = t2_eV * eV_to_J
    M_J = M_eV * eV_to_J

    h_k_h, eigsys_h = make_haldane_hamiltonian_oriented(
        a1, a2, deltas, tJ, t2J, phi, M_J
    )

    F_bz_h = berry_curvature_bz(eigsys_h, b1, b2, Nk=Nk, band_index=0)
    # normalisasi: BZ efektif satu fundamental → 2π
    chern_h = jnp.sum(F_bz_h) / (2.0 * jnp.pi)
    return F_bz_h, chern_h

# ============================================================
# DEMO / CHECKS
# ============================================================

if __name__ == "__main__":
    t_eV = 2.7
    tJ = t_eV * eV_to_J
    eps_diag = 0.0

    # Titik K analitik untuk choice a1, a2 ini:
    # salah satu K = (b1 + 2 b2) / 3
    K0 = (b1 + 2.0 * b2) / 3.0
    K_ref, iters = newton_refine_dirac(K0, tJ, deltas, eps_diag)

    print("====================================")
    print("   NEWTON REFINEMENT — GRAPHENE")
    print("====================================")
    print("K0           :", K0)
    print("K_refined    :", K_ref)
    print("iters        :", iters)
    print("====================================")

    # vF fit
    vF_fit, spread = fit_vF_ring(K_ref, tJ, deltas, eps_diag, a)
    vF_analytic = 3.0 * tJ * a / (2.0 * hbar)
    print("====================================")
    print("   RING vF FIT — GRAPHENE")
    print("====================================")
    print("vF_analytic      :", float(vF_analytic))
    print("vF_fit           :", float(vF_fit))
    print("vF_fit / analytic:", float(vF_fit / vF_analytic))
    print("spread / analytic:", float(spread / vF_analytic))
    print("====================================")

    # Berry phase di K (massless)
    h_k0 = lambda k: h_graphene(k, tJ, deltas, eps_diag)
    eigsys0 = lambda k: jnp.linalg.eigh(h_k0(k))
    _, gamma = berry_phase_ring(K_ref, eigsys0, a, q_abs_rel=5e-3, directions=512, band_index=0)
    print("====================================")
    print("   BERRY PHASE — GRAPHENE (AROUND K)")
    print("====================================")
    print("gamma (raw)      :", float(gamma))
    print("gamma mod 2π     :", float((gamma + jnp.pi) % (2.0 * jnp.pi) - jnp.pi))
    print("gamma / π        :", float(gamma / jnp.pi))
    print("====================================")

    # QWZ scan
    print("====================================")
    print("   QWZ CHERN INSULATOR — CHERN vs m")
    print("====================================")
    for m in [-3.0, -2.0, -1.0, 0.0, 1.0]:
        F_q, ch_q = qwz_bz_map_and_chern(m, Nk=61)
        print(f"m = {m: .1f} | Chern ≈ {float(ch_q): .4f}")
    print("====================================")

    # Haldane scan
    print("====================================")
    print("   HALDANE (ORIENTED NNN) — CHERN vs M")
    print("====================================")
    for M in [-0.5, 0.0, 0.5]:
        F_h, ch_h = haldane_bz_map_and_chern_oriented(
            a1, a2, deltas, b1, b2,
            t_eV=1.0, t2_eV=0.1, phi=jnp.pi/2, M_eV=M, Nk=61
        )
        print(f"M = {M: .2f} eV | Chern ≈ {float(ch_h): .4f}")
    print("====================================")
