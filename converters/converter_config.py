from builder import CONVERTERS
converter1_cfg = dict(type='Converter1', a=1,b=2)
converter2_cfg = dict(type='Converter2', a=3,b=4)

converter1 =CONVERTERS.build(converter1_cfg)
print(converter1)
converter2 =CONVERTERS.build(converter2_cfg)
print(converter1)
