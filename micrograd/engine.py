class Value:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        return Value(self.data + other.data)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        return Value(self.data * other.data)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
