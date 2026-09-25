import pytest
import torch
from embeddings import RoPE, TritonRoPE, rope_forward, rope_backward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
T_MAX = 256
TOL = {
    torch.float32: dict(atol=2e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=3e-2, rtol=2e-2),
}


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def offset_for(kind, T):
    return {"off0": 0, "off7": 7, "offend": T_MAX - T}[kind]


def make(shape, dtype, layout):
    """Tensor with logical shape (B, n, T, d) and a chosen memory layout."""
    B, n, T, d = shape
    kw = dict(device=DEV, dtype=dtype)
    if layout == "contig":
        t = torch.randn(B, n, T, d, **kw)
    elif layout == "heads_T":        # transpose of dims 1,2: last stride still 1
        t = torch.randn(B, T, n, d, **kw).transpose(1, 2)
    elif layout == "last_T":         # last stride != 1 -> contiguous() path
        t = torch.randn(B, n, d, T, **kw).transpose(-1, -2)
    elif layout == "expand_row":     # outer strides 0, last stride 1
        t = torch.randn(d, **kw).expand(B, n, T, d)
    elif layout == "expand_all":     # all strides 0 (what .sum() produces)
        t = torch.randn((), **kw).expand(B, n, T, d)
    else:
        raise ValueError(layout)
    assert t.shape == (B, n, T, d)
    return t


common = [
    pytest.mark.parametrize("B,n", [(1, 1), (2, 3)], ids=["B1n1", "B2n3"]),
    pytest.mark.parametrize("T", [1, 5, 128], ids=lambda v: f"T{v}"),
    pytest.mark.parametrize("d", [2, 64, 96], ids=lambda v: f"d{v}"),
    pytest.mark.parametrize("off_kind", ["off0", "off7", "offend"]),
    pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]),
]


def apply_common(f):
    for m in reversed(common):
        f = m(f)
    return f


# ---------- 1. match the reference ----------

@apply_common
@pytest.mark.parametrize("layout", ["contig", "heads_T", "last_T"])
def test_forward_matches_reference(B, n, T, d, off_kind, dtype, layout):
    off = offset_for(off_kind, T)
    x = make((B, n, T, d), dtype, layout)
    x_before = x.clone()
    ref = RoPE(d, T_MAX).to(DEV)
    tri = TritonRoPE(d, T_MAX).to(DEV)

    out_ref = ref(x, off)
    out_tri = tri(x, off)

    assert out_tri.shape == x.shape and out_tri.dtype == x.dtype
    torch.testing.assert_close(out_tri, out_ref, **TOL[dtype])
    torch.testing.assert_close(x, x_before, atol=0, rtol=0)  # input not mutated


@apply_common
@pytest.mark.parametrize("g_layout", ["contig", "heads_T", "last_T", "expand_row", "expand_all"])
def test_backward_matches_reference(B, n, T, d, off_kind, dtype, g_layout):
    off = offset_for(off_kind, T)
    x = make((B, n, T, d), dtype, "contig")
    g = make((B, n, T, d), dtype, g_layout)
    ref = RoPE(d, T_MAX).to(DEV)
    tri = TritonRoPE(d, T_MAX).to(DEV)

    x_ref = x.detach().clone().requires_grad_()
    x_tri = x.detach().clone().requires_grad_()
    (dx_ref,) = torch.autograd.grad(ref(x_ref, off), x_ref, g)
    (dx_tri,) = torch.autograd.grad(tri(x_tri, off), x_tri, g)

    assert dx_tri.shape == x.shape and dx_tri.dtype == x.dtype
    torch.testing.assert_close(dx_tri, dx_ref, **TOL[dtype])


# ---------- 2. self-consistency (independent of the reference) ----------

@pytest.mark.parametrize("d", [2, 64, 96], ids=lambda v: f"d{v}")
@pytest.mark.parametrize("off_kind", ["off0", "off7", "offend"])
def test_backward_is_adjoint_of_forward(d, off_kind):
    # <R x, g> == <x, R^T g>  holds iff backward is the exact transpose of forward
    B, n, T = 2, 3, 37
    off = offset_for(off_kind, T)
    x = make((B, n, T, d), torch.float32, "contig")
    g = make((B, n, T, d), torch.float32, "contig")

    lhs = (rope_forward(x, off).double() * g.double()).sum()
    rhs = (x.double() * rope_backward(g, off).double()).sum()
    scale = (x.double().norm() * g.double().norm()).item()
    assert abs(lhs - rhs).item() <= 1e-5 * scale, (lhs.item(), rhs.item())


@pytest.mark.parametrize("d", [2, 64, 96], ids=lambda v: f"d{v}")
@pytest.mark.parametrize("off_kind", ["off0", "off7", "offend"])
def test_backward_inverts_forward(d, off_kind):
    # R is orthogonal: R^T R x == x
    B, n, T = 2, 3, 37
    off = offset_for(off_kind, T)
    x = make((B, n, T, d), torch.float32, "contig")
    x_rt = rope_backward(rope_forward(x, off), off)
    torch.testing.assert_close(x_rt, x, atol=1e-5, rtol=1e-5)


# ---------- 3. module contract ----------

def test_offset_past_T_max_raises():
    tri = TritonRoPE(64, T_MAX)
    x = torch.randn(1, 1, 8, 64, device=DEV)
    with pytest.raises(AssertionError):
        tri(x, T_MAX - 7)