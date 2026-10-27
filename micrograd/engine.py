class Value:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        return Value(self.data + other.data)
    
    def __mul__(self, other):
        return Value(self.data * other.data)
    