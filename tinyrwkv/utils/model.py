from tinygrad.state import get_parameters


def compile_net(run, special_names):
    # functions that run the net
    functions = {}
    bufs = {}
    bufnum = 0
    byte_offset = 0
    buf_offsets = {}
    statements = []
    bufs_to_save = {}
    for fxn, args in run.jit_cache:
        functions[fxn.name] = fxn.prg

        cargs = []
        for i, arg in enumerate(args):
            key = id(arg)
            if key not in bufs:
                if key in special_names:
                    bufs[key] = (special_names[key], len(arg._buf), arg.dtype)
                else:
                    bufs[key] = (f"scratch_{bufnum}", len(arg._buf), arg.dtype)
                    bufnum += 1
                    if i > 0:
                        bufs_to_save[bufs[key][0]] = arg

                        # offset into weights
                        if key not in buf_offsets:
                            buf_offsets[key] = byte_offset
                            byte_offset += len(arg._buf) * (
                                4 if str(arg.dtype)[7:] == "float" else 2
                            )

            # use offset into weights
            if key in special_names or bufs[key][0] not in bufs_to_save:
                cargs.append(bufs[key][0])
            else:
                cargs.append(
                    f"({str(bufs[key][2])[7:]}*)(tinyrwkv->weights + {buf_offsets[key]})"
                )

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
