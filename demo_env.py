
import math

class Value():

    def __init__(self, data, _prev=[], _op='', label=''):
        self.data = data
        self.children =_prev
        self._op = _op
        self.label = label
        self._backward= lambda : None
        self.grad = 0.0

    def __repr__(self):
        return f'Value({self.data}, {self._op})'
    

    def __add__(self, other):
        other=other if isinstance(other, Value) else Value(other)
        out= Value(self.data + other.data, _prev=[self, other], _op='+')
        def _backward():
            self.grad += 1.0 * out.grad    
            other.grad += 1.0 * out.grad
        out._backward=_backward
        return out
    def __radd__(self,other):
        return self+other
    
    def __neg__(self):  # -self
        return self*(-1)
    
    def __sub__(self,other ):   #self-other
        return self +(-other)

    def __mul__(self, other):
        other= other if isinstance(other, Value) else Value(other)
        out= Value(self.data * other.data, _prev=[self, other], _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data *out.grad
        out._backward= _backward
        return out
    def __rmul__(self, other):
        return self*other
    
    def __truediv__(self, other):   #self/other
        return self*other**-1
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
    
    def backward(self):
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

        
        

a= Value(2.0, label='a')
b= Value(3.0, label = 'b')
c=a+b; c.label='c'
d= c+1; d.label='d'
e= d*a; e.label='e'
f= e.tanh()
print(e, f)
# '''
# # print('The children of e are as follows: ')
# # for child in e.children:    
# #     print('\t', child.label, '==>', child.data)
# # print('The operation of e is ', e._op)

# # testing lines
# # topo(e)
# # print([x.label for x in topolist])
# # e.grad=1.0
# # e._backward()
# # print(e.grad)
# # print(d.grad)
# # print(a.grad)
# # d._backward()
# # print(c.grad)
# # print(a.grad)
# # c._backward()
# # print(a.grad)
# '''


# f.backward()
# print(f.grad)
# print(d.grad)
# print(a.grad)

class Neuron():
    def __init__(self, nin):
        self.w= [Value(random.uniform(-1,1) for i in range(nin))]
        self.b= Value(random.uniform(-1,1))
    def __call__(self, x):
        act= sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out= act.tanh()
        return out
    def parameters(self):
        return self.w +[self.b]

class Layer():
    def __init__(self, nin, nout):
        self.neurons= [Neuron(nin) for i in range(nout)]
    
    def __call__(self, x):
        outs= [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz= [nin] + nouts 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x= layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
