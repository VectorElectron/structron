from .memory import TypedMemory
from .linear import TypedDeque, TypedStack, TypedQueue
from .heap import TypedHeap
from .hash import TypedHash
from .rbtree import TypedRBTree
from .avltree import TypedAVLTree
from .graph import TypedGraph

def register(name, region=globals()):
    temp = '''
    class %sStruct(nb.types.StructRef): pass
    class %sProxy(structref.StructRefProxy): pass
    '''%(name, name)
    exec('\n'.join([i.strip() for i in temp.split('\n')]), region)
    return region['%sStruct'%name], region['%sProxy'%name]

