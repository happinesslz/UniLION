# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def MambaInnerFnNoOutProj_flop_jit(inputs, outputs):
    # import ipdb; ipdb.set_trace()
    # print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    D = D // 2
    # N = inputs[2].type().sizes()[1]
    N = 16 #  A (D, N) default N = 16
    
    # https://github.com/state-spaces/mamba/issues/110
    # selective scan 9 * b * d * l
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=True)
    
    # causual conv1d k * b * d * l
    flops += 4 * B * D * L * 2
    
    # x_proj && dt_proj included in selective scan
    # # x_proj b * l * d * (16 * 2 + [D / 16])
    # flops += B * L * D * (16 * 2 + (D // 16))
    
    # # dt_proj b * l * [D / 16] * [D / 16]
    # flops += B * L * (D // 16) * (D // 16)
    
    # in_proj : B * L * 64 * 256 
    flops += B * L * (D / 2) * (D * 4) * 2

    # out_proj : B * L * 64 * 128
    flops += B * L * (D / 2) * D * 2
    
    return flops