from contextlib import redirect_stdout
from io import StringIO
from pprint import pprint
import time
from functools import wraps
""" profile memory footprint with @mem """
# from memory_profiler import profile as mem
import json


class cases:
    """ from key, val arg, add/append to dictionary"""

    def __str__(self):
        return str(self.mem)

    def __len__(self):
        return len(self.mem)

    def __repr__(self):
        return str(self.mem)

    def __iter__(self):
        return iter(self.mem)

    def keys(self):
        return list(self.mem.keys())

    def __call__(self, k=None, v=None):
        if not k: return self.mem
        check_iter=lambda x: (v) if not hasattr(v, '__iter__') else v

        if k not in self.mem:
            print(v)
            self.mem[k] = set(check_iter(v))
        else:
            try:
                if v not in self.mem[k]:
                    self.mem[k].add(check_iter(v))
            except AttributeError:
                self.mem[k].append(check_iter(v))
        return self.mem

    def to_hashable(self):
        for k, v in self.mem.items():
            try:
                self.mem[k] = sorted(list(v))
            except TypeError:
                print('dang')
                raise Exception(
                    'TypeError Key:{}, Val {} '.format(k, v)
                )
        return self.mem

    def __init__(self):
        self.mem = {}

    @staticmethod
    def fromKeys(keys):
        x = cases()
        x.mem = {k: set() for k in keys}
        return x

    # def drop_all():
    #     cases.mem = {}

    # def copy():
    #     from copy import copy
    #     return copy(cases)


def writer(f, silent=False):

    def wrapper(*args, **kwargs):
        try:
            _j, name = f(*args)
        except TypeError:
            return (pretty(args))

        j = json.dumps(
            _j, indent=4, sort_keys=True, ensure_ascii=False, *kwargs
        )
        if not name.endswith('.json'):
            name += '.json'
        with open(name, 'w') as obj:
            obj.write(j)
        return 'JSON writer: saved {} to file'.format(name)

    return wrapper


def reader(f):

    def wrapper(*args):
        _json = f(*args)
        if not _json.endswith('.json'):
            _json += '.json'
        return json.load(open(_json))

    return wrapper


@reader
def get_json(_j):
    return _j


@writer
def set_json(_j, name):
    return _j, name


class Count:
    """ Call this to increment. Returns current value """
    n = 0

    def __new__(self):
        Count.n += 1
        return Count.n - 1


class catcher:
    """ One stop logging object """
    from collections import defaultdict
    items = defaultdict(list)

    def __call__(self, item, name='inner'):
        try:
            catcher.items[name].append(item)
        except:
            Exception(item, 'is list')

    def reset(self, name='inner'):
        catcher.items['inner'] = []

    def show(self, name='inner'):
        pretty(catcher.items[name])

    def __str__(self):
        pretty(catcher.items['inner'])
        return '\n'


def ez_catch(func_or_lambda, args):
    try:
        func_or_lambda(args)
    except Exception as e:
        print(e)
        pass


def pretty(json_mappable, export=False, show_type=False):
    from beeprint import pp
    if export:
        pp(json_mappable, output=False)
    else:
        pp(json_mappable, output=True)


def merge_jsons(jsonList, show_count=False):
    from jsonmerge import merge

    if len(jsonList) < 1:
        raise Exception('jsonList must be more than one item')
    try:
        data = jsonList[0]
        deltas = []
        count = lambda x: deltas.append(len(str(x)))

        for j in jsonList[1:]:
            count(data)
            data = merge(data, j)
        count(data)
        if show_count:
            delta = deltas[-1] - deltas[0]
            if delta > 0:
                print('Merged JSON added features {}'.format(delta))
        return data
    except TypeError:
        print("Json only bitches")


def debug(*args, export=False):
    """ Pretty print with a cyanide pill """

    json_mappable = args
    pretty(json_mappable)
    print('\n')
    exit("(((exit)))")


class json_collector:
    ''''collects all of the type field outputs for json file write'''

    def __init__(self, name='meta_all_'):
        self.heap = []
        self.name = name

    def update(self, dict_):
        if not self.heap:
            self.heap = {}
        self.heap.update(dict_)

    def __len__(self):
        return len(str(self.heap))

    @writer
    def commit(self):

        return self.heap, self.name


def timeit(func):
    """ Returns time of delta for function in seconds """

    @wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        delta = round((te - ts), 1)
        if delta > 1:
            t = ' '.join([str(delta), 's'])
        else:
            t = ' '.join([str(round((te - ts) * 1000, 1)), 'ms'])

        print('Function', func.__name__, 'time:', t)
        return result

    return timed


def pandas_print(func):
    """ Renders all DataFrames yielded in Jupyter Notebook cell """
    import pandas as pd
    from IPython.core import display as ICD

    @wraps(func)
    def wrapper(*args):
        try:
            for f in func(*args):
                if isinstance(f, pd.core.frame.DataFrame):
                    ICD.display(f)
                # elif isinstance(f, list):
                #     for sub in f:
                #         print(sub)
                elif f == None:
                    print('\n')
                else:
                    print(f)

        except Exception as e:
            print('ERROR!', e, e.__doc__)
            return

    return wrapper


def silence(func):
    """ 
    Silences print statements by directing to a return statement.
    Also returns anything returned by the function. 
    Returns ("print buffer","return from original function")
    """

    @wraps(func)
    def wrapper(*args):
        with StringIO() as buf, redirect_stdout(buf):
            value = ''
            value = func(*args)
            print('redirected', func.__name__)
            return buf.getvalue(), value

    return wrapper


def flatList(func):
    """ 
    Flatten arbitrarily nested lists while perserving ordinality:
    [A,B,C,[[D],E]] --> [A,B,C,D,E] 
    """

    def flatland(nested):
        for sub in nested:
            if isinstance(sub, list):
                yield from flatland(sub)
            else:
                yield sub

    @wraps(func)
    def wrapper(*args):
        x = func(*args)
        return list(flatland(x))

    return wrapper


def lazy_pprint(func):
    ''' For transparent functions/iterators. PrettyPrint to stdout with just yield/return '''

    @wraps(func)
    def wrapper(*args):
        for f in func(*args):
            pprint(f)
            yield f

    return wrapper


def lazy_print(func):
    ''' For transparent functions/iterators. Print to stdout with just yield/return '''

    @wraps(func)
    def wrapper(*args):
        for f in func(*args):
            print(f)
            yield f

    return wrapper


import pickle


def brine(func):
    ''' Pickle func result, with function as default name or dec arg '''

    @wraps(func)
    def wrapper(*args):
        fileName = func.__name__ + '.pkl'
        with open(fileName, 'wb') as file:
            pickle.dump(func(*args), file)
        print('pickled', repr(fileName))

    return wrapper
