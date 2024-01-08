
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
    
    def tanh(self):
        x= self.data
        # out = Value(math.exp(2*x)-1)/(math.exp(2*x) + 1)
        t = math.tanh(x)
        out=Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward=_backward
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


f.backward()
print(f.grad)
print(e.grad)



