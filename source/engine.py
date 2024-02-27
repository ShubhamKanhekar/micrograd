
import math

class Value():

    def __init__(self, data, _prev=[], _op='', label=''):
        self.data = data
        self.children =set(_prev)
        self._op = _op
        self.label = label
        self._backward= lambda : None
        self.grad = 0.0

    def __repr__(self):
        return f'Value({self.data}, grad= {self.grad})'
    
    def __gt__(self, other):
        out= self.data> other.data
        def _backward():
            self.grad +=out.grad
            other.grad += out.grad
        out._backward= _backward
        return out
    
    def __add__(self, other):
        other=other if isinstance(other, Value) else Value(other)
        out= Value(self.data + other.data, _prev=[self, other], _op='+')
        def _backward():
            self.grad += out.grad    
            other.grad += out.grad
        out._backward=_backward
        return out
    def __radd__(self,other):
        return self+other
    
    def __neg__(self):  # -self
        return self*(-1)
    
    def __sub__(self,other ):   #self-other
        return self + (-other)
    def __rsub__(self,other):   # other - self
        return other + (-self)

    def __mul__(self, other):
        other= other if isinstance(other, Value) else Value(other)
        out= Value(self.data * other.data, _prev=[self, other], _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data *out.grad
        out._backward= _backward
        return out
    def __rmul__(self, other):      #other * self
        return self*other
    
    def __truediv__(self, other):   #self/other
        return self*other**-1
    def __rtruediv__(self,other):   # other / self
        return other* self**(-1)
    ''' we can use __pow__() to divide too : a/b = a*(1/b) = a*(b**-1)
     we have to redefine division above in the method __truediv__(): 
     we will also implement __pow__() function. we do this for powers other than Value objects viz: ints and floats.
     refer video: https://youtu.be/VMj-3S1tku0?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=5596'''
    
    def __pow__(self, other):  #self**other (other is a number and not a Value obj)
        assert isinstance(other,(int,float)), 'only supporting int/float powers for now '
        out= Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            # d(x**n)/dx= n(x**(n-1))
            self.grad += (other * self.data**(other -1))* out.grad
        out._backward= _backward
        return out


    def tanh(self):
        x= self.data
        # out = Value(math.exp(2*x)-1)/(math.exp(2*x) + 1)
        t = math.tanh(x)
        out=Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward=_backward
        return out
    
    def exp(self):
        x= self.data
        out = Value(math.exp(x), (self, ) , 'exp')
        def _backward():
            self.grad += out.data * out.grad     #since e**x 's derivative is same i.e. e**x
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, _prev= (self,), _op='ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward= _backward
        return out
    
    def backward(self):
        # create topological order
        topo=[]
        visited= set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad= 1.0
        for node in reversed(topo):
            node._backward()

        
        



