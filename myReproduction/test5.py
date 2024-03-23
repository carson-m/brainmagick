import numpy as np

a = np.array([True, False, True])
c = np.array([False, True, False])
b = np.array([1,1,1])
b[c] = 0
b[a] = np.array([2,3])
print(b)
d = np.empty(3,dtype=bool)
print(d)

class test:
    def set_a(self):
        self.a = 3
        
obj = test()
#print(obj.a)
obj.set_a()
print(obj.a)

print(len(np.ones((5, 2))))