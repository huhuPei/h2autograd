# h2autograd
此版本是基于micrograd(https://github.com/karpathy/micrograd)进行二次开发

微分机制特点：
1、通过定义Value对象，用做反向传播的图节点，该节点实现了基本的加法与乘法计算，同时会保存该节点的梯度；
2、将多个Value对象通过reshape为(r, c)形状得到Tensor对象；
3、通过Tensor对象是实现深度全连接网络。