To run this project, try

```shell
bash run.sh
```

This is a repo for the HTML5 project in the physics experiment course

# 网站功能

实现任意二维图样的衍射

- 功能
  - 支持用户上传图片
    - 后端处理为灰度图并满足分辨率要求
  - 支持调节衍射参数
  - 支持在线绘制图片
- 教程
  - 加入此setting的演示图片
  - 加入算法介绍


# 计划

- 第一周
  - 确定网站功能
  - 后端模拟
    - 学习物理和算法
- 第二周
  - 前端
    - 实现框架
  - 后端
    - 学习物理和算法
    - 实现部分功能
- 第三周
  - 前端
    - 和后端对接
    - 讨论可能的新功能
  - 后端
    - 实现所有功能
    - 初步测试
- 第四周
  - 测试
    - 经典算例: Fraunhofer Diffraction
    - 前段测试各个功能
- 第五周
  - 润色网站视觉效果
  - 添加算法介绍
  - Presentation

## Algorithm

### Angular Spectrum Method

We employed the angular spectrum method to simulate the diffraction where the 
input and the output are both in a 2-dimensional plane.

So the direction of the EM wave propagation is denoted as z whereas the input and output
plane is denoted as x-y plane. 

Let's write down the scalar Helmholtz equation for E in free space:
$
(\nabla^2 + k^2) E(x, y, z) = 0
$, 
where $k$ is the wave vector ($k=\frac{2\pi}{\lambda}$). The Fourier transform of 
the two sides of the above equation yields 
$
\int_{x, y} (\nabla^2 + k^2) E(x, y, z) e^{ik_x x} e^{ik_y y}dx dy = 0, 
$
which simplifies into 
$
(\frac{\partial }{\partial z^2} + k^2 - k_x^2 - k_y^2) \tilde{E}(k_x, k_y, k_z) = 0. 
$

By **denoting** $k_z\coloneqq k^2- k_x^2-k_y^2$, we get a simpler form of
$
(\frac{\partial }{\partial z^2} + k_z^2) \tilde{E}(k_x, k_y, k_z) = 0. 
$

Essentially, this is a one-dimensional wave equation in the **Fourier space** which depends only on $k_z$. Its solution is simply
$
\tilde{E}(k_x, k_y, k_z) = \tilde{E}(k_x, k_y, 0)e^{ik_zz}, 
$
which is to say, in the Fourier domain whose corresponding real space is the x-y plane, the wave propagation is as simple as a **multiplication**.

In another word, after acquiring the wave on the plane of $z=0$, we can first FT the x-y dimension of the wave $E(x, y, z)$, multiply the $e^{ik_z z}$ to it and then inverse FT it back to the real space. In this way, we can efficiently get the output ($E(x,y,z_0)$ for some $z_0$ plane) based on the input ($E(x,y,0)$).

