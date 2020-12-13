import torch
import numpy as np
# a = torch.rand(4, 2)
# print(a)
#
# idx = 1
# a = a[torch.arange(a.size(0))!=1]
# print(a)

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
arr = np.delete(arr, [1,2], axis=0)
print(arr)