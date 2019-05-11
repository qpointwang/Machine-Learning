# 代码内容来自 https://github.com/zergtant/pytorch-handbook

from __future__ import print_function
import torch

# Tensors与Numpy中的 ndarrays 类似，但是在PyTorch中 Tensors 可以使用GPU进行计算.

# 创建一个 5x3 矩阵, 但是未初始化:
x = torch.empty(5, 3)
print(x)
# 创建一个随机初始化的矩阵:
x = torch.rand(5, 3)
print(x)
# 创建一个0填充的矩阵，数据类型为long:  float,float16,float32,float64.int,long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
# 创建tensor并使用现有数据初始化:
x = torch.tensor([5.5, 3])
print(x)
x = torch.tensor([[5.5, 3],[2, 5]])
print(x)
# 根据现有的张量创建张量。 这些方法将重用输入张量的属性，例如， dtype，除非设置新的值进行覆盖
x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
print(x)
x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
print(x)                                      # 对象的size 是相同的，只是值和类型发生了变化
# 获取 size
# 译者注：使用size方法与Numpy的shape属性返回的相同，张量也支持shape属性，后面会详细介绍
print(x.size())

# 加法
y = torch.rand(5, 3)
print(x + y)
# 加法2
print(torch.add(x, y))
# 提供输出tensor作为参数，将结果赋值给result
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# 替换
# Note:任何 以``_`` 结尾的操作都会用结果替换原变量. 例如: ``x.copy_(y)``, ``x.t_()``, 都会改变 ``x``.
print(y)
y.add_(x)
print(y)
# 可以使用与NumPy索引方式相同的操作来进行对张量的操作
print(x)
print(x[0, :])
print(x[:, 1])

# torch.view: 可以改变张量的维度和大小
# 译者注：torch.view 与Numpy的reshape类似
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  #  size -1 从其他维度推断   -1相当于不指定，根据16/8来进行判断
print(x.size(), y.size(), z.size())
# 如果你有只有一个元素的张量，使用.item()来得到Python数据类型的数值
x = torch.randn(1)
print(x)
print(x.item())
# 更多操作 https://pytorch.org/docs/torch
# NumPy转换
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b) # 这里b也变了
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
# 所有的 Tensor 类型默认都是基于CPU， CharTensor 类型不支持到 NumPy 的转换.

# CUDA 张量
# 使用.to 方法 可以将Tensor移动到任何设备中
# is_available 函数判断是否有cuda可以使用
# ``torch.device``将张量移动到指定的设备中
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
    print(y)
    x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
    print(x)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改


# Autograd: 自动求导机制
# PyTorch 中所有神经网络的核心是autograd包。我们先简单介绍一下这个包，然后训练第一个简单的神经网络。
# autograd包为张量上的所有操作提供了自动求导。
# 它是一个在运行时定义的框架，这意味着反向传播是根据你的代码来确定如何运行，并且每次迭代可以是不同的。

print('------------------------------------------------------------\n')

x = torch.ones(2, 2, requires_grad=True)
print(x)
# 对张量进行操作:
y = x + 2
print(y)
# 结果y已经被计算出来了，所以，grad_fn已经被自动生成了。
print(y.grad_fn)
# 对y进行一个操作
z = y * y * 3
out = z.mean()
print(z, out)
# .requires_grad_( ... ) 可以改变现有张量的 requires_grad属性。
# 如果没有指定的话，默认输入的flag是 False。
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
print(a)
b = (a * a).sum()
print(b)
print(b.grad_fn)

# 梯度
# 反向传播 因为 out是一个纯量（scalar），out.backward() 等于out.backward(torch.tensor(1))
print(out)
print(out.backward())
print(x)
print(y)
print(x.grad)


# x = torch.tensor([[5.5, 3],[2, 5]], requires_grad=True)
# print(x)
# y = x + 2
# print(y)
# #z = y * y * 3
# #out = y.mean()
# #print(out)
# y.backward()
# print(x.grad)

# 可以使用 autograd 做更多的操作
x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
	print((x ** 2).requires_grad)