class Value:
    def __init__(self, data, children=(), op=""):
        self.data = data
        self._children = set(children)
        self._op = op
        self.gradient = 0.0
        self._backward = lambda: None

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build_topo(c)
                topo.append(v)

        build_topo(self)
        self.gradient = 1.0
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.gradient += 1.0 * out.gradient
            other.gradient += 1.0 * out.gradient

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.gradient += other.data * out.gradient
            other.gradient += self.data * out.gradient

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Value({self.data})"
