import importlib

def reload(m):
    try:
        m
    except NameError:
        import m
    else:
        importlib.reload(m)
