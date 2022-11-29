from builder import CONVERTERS

@CONVERTERS.register_module()
class Converter1:
    def __init__(self, a, b):
        self.a = a
        self.b = b
