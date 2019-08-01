import inspect
import marshal
import base64

# Helper
trim_name = lambda x: 'x' + x.name.rsplit('/', 1)[0].rsplit(':', 1)[0]
trim_names = lambda x: '[' + ', '.join([trim_name(i) for i in x]) + ']'

def update_argument(f, replace=None):
    li = [l.strip() for l in f.split('\n')]
    ag = li[0].strip('):').split('(')[1].split(',')
    if replace and isinstance(replace, Iterable):
        r = [a.split('=')[0] for a in ag][:-len(replace)]
        ags = r + list(replace)
        return 'lambda '+ ', '.join(r) + ': ' + \
            li[0].split('def ', 1)[1].split('(', 1)[0] + '(' +', '.join(map(str,ags)) + ')'
    else:
        ags = [a.split('=')[0] for a in ag]
        return li[0].split('(')[0] + '(' + ', '.join(map(str,ags)) + '):'

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

# Main
lines = []
functions = []
for l in k_model.layers:
    
    # Get Basic Informations
    cls = 'layers.' + l.__class__.__name__
    out = trim_name(l.output)
    conf = l.get_config()
    conf = {key: c for key, c in conf.items() if key!='trainable'} #if c
    
    # Handle InputLayer
    if l.__class__.__name__ == 'InputLayer':
        shape = conf['batch_input_shape'][1:]
        line = "{out} = layers.Input(shape={shape}, name='{name}')".format(name=l.name,shape=shape, out=out)
        lines.append(line)
        continue
    
    # Decode Lambda
    if l.__class__.__name__ == 'Lambda':
        code = marshal.loads(base64.b64decode(conf['function'][0].encode('raw_unicode_escape')))
        code_readable = inspect.getsource(code).strip()
        arguments = conf['function'][1]
        if code_readable.startswith('def '):
            functions.append(code_readable)
        conf['function'] = update_argument(code_readable, arguments)
        del conf['function_type']
        del conf['output_shape_type']
        del conf['arguments']
    
    # Get LayerInputs
    if 'input' in dir(l):
        inp = trim_names(l.input) if isinstance(l.input, list) else trim_name(l.input)
    else:
        inp = trim_names(l.inputs)
    
    # Simplify Initializers
    for k, v in list(conf.items()):
        if k in ['kernel_initializer', 'bias_initializer'] and isinstance(v, dict) \
            and list(v.keys()) == ['class_name', 'config'] and not v['config']:
            conf[k] = v['class_name'].lower()
    
    # Escape Strings
    for k, v in list(conf.items()):
        if any([isinstance(v, t) for t in [type(None), float, int, type(()), dict, list]]) or v.startswith('lambda '):
            continue
        conf[k] = "'" + conf[k] + "'"
    
    
    # Remove default arguments
    params = ', '.join(['='.join(map(str,i)) for i in conf.items() 
                        if not (i[0] in get_default_args(l.__class__) and \
                                get_default_args(l.__class__)[i[0]]==i[1])])
    line = '{out} = {class_name}({params})({inp})'.format(class_name=cls, params=params, inp=inp, out=out)
    lines.append(line)

print('from keras import layers\nfrom keras import backend as K\n')
for f in set(functions):
    li = [l.strip() for l in f.split('\n')]
    print(update_argument(li[0]))
    print('\n'.join(['    '+l for l in li[1:]]))
print('\n')
print('\n'.join(lines))
