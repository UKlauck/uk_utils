def layer_size(layer):
    size = 1
    los = layer.output_shape
    if type(los) == list:
        los = los[0]
    for s in los:
        if s is not None:
            size *= s

    return size


def no_params(layer):
    l_type = layer.__class__.__name__
    no_params = 0

    if l_type.lower().find('conv2d') != -1:
        nf = layer.get_config()['filters']
        ksize = layer.get_config()['kernel_size']
        no_params = nf * (ksize[0] * ksize[1] * layer.input_shape[-1] + 1)
    elif l_type.lower().find('dense') != -1:
        no_params = layer.output_shape[-1] * (layer.input_shape[-1] + 1)
    if l_type.lower().find('pooling') != -1:
        pass
    elif l_type.lower().find('flatten') != -1:
        pass

    return no_params


def print_model_summary(model):
    total_params = 0
    sep = '_' * 56

    print(sep)
    print('%.15s%s' % ('Layer', ' '*10), end='')
    print('Tensor size          ', end='')
    print('%10s' % ('Size'), end='')
    print('%10s' % ('# params'), end='')
    print('')
    print(sep)

    for l in model.layers:
        l_type = l.__class__.__name__
        l_shape = str(l.output.shape)
        l_size = layer_size(l)
        l_params = no_params(l)
        total_params += l_params

        print('%.15s%s' % (l_type, ' ' * (15 - len(l_type))), end='')
        print(l_shape, ' ' * (20 - len(l_shape)), end='')
        print('%10d' % (l_size), end='')
        print('%10d' % (l_params), end='')
        print('')

    print(sep)
    print('%56d' % (total_params))


def print_summary(model):
    print(model.summary())
    return


def save_summary(model, filename):
    with open(filename, 'w') as fd:
        fd.write('\\tiny{\\begin{verbatim}\n')
        fd.write(str(model.summary()))
        fd.write('\n\\end{verbatim}}')
    return


def print_summary_tables(model, tables=(0, 1, 2)):
    for t in tables:
        print(model.summary().tables[t])
    return


def save_summary_tables(model, tables=(0, 1, 2), filename=None):
    if filename is None: return
    with open(filename, 'w') as fd:
        fd.write('\\tiny{\\begin{verbatim}\n')
        for t in tables:
            fd.write(str(model.summary().tables[t]))
        fd.write('\n\\end{verbatim}}')
    return
