from tinygrad.nn.optim import get_parameters


def compile_net(run, special_names):
    # functions that run the net
    functions = {}
    bufs = {}
    bufnum = 0
    statements = []
    bufs_to_save = {}
    for fxn, args in run.jit_cache:
        functions[
            fxn.name
        ] = fxn.prg  # NOTE: this assumes all with the same name are the same
        cargs = []
        for i, arg in enumerate(args):
            key = id(arg)
            if key not in bufs:
                if key in special_names:
                    bufs[key] = (special_names[key], len(arg._buf), arg.dtype)
                else:
                    bufs[key] = (f"buf_{bufnum}", len(arg._buf), arg.dtype)
                    bufnum += 1
                    if i > 0:
                        bufs_to_save[
                            bufs[key][0]
                        ] = arg  # if first usage of a buffer is not an output, and it's not a special name
            cargs.append(bufs[key][0])
        statements.append(f"{fxn.name}({', '.join(cargs)});")

    return functions, statements, bufs, bufs_to_save


def count_parameters(model):
    params = get_parameters(model)
    count = 0
    for p in params:
        param_count = 1
        for s in p.shape:
            param_count *= s
        count += param_count
    return count
