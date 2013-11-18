class X(object):
    
    def __init__(self,a):
        self.a = a          


def f(x):
    inst = X(x)
    return inst

inst = f(0)
print inst.a
f(inst)
print inst.a

inst = X(0)
print inst.a

from multiprocessing import pool
p = pool.Pool(2)
list = p.map(f,[1,2,3])

for instance in list:
    print instance.a
    
