# Mean-Flownav-Note

主要的文件在 `meanflow`文件夹里面，预计第一阶段训练就是正常把flownav的训练跑一遍（或者如果不跑flownav也可以直接拿flownav已有的权重，等后面做微调即可）然后第二阶段只需把loop里面的train函数和eval函数换成meanflow文件夹里面定义的函数，再训练微调即可即可。

`meanflow.train`是从 `flownav.train`拷贝过来的，主要的改动在于meanflow的训练数据来源应该是flownav生成的数据，即1-rectified的数据

以及训练损失的计算应该遵循meanflow的计算方式

但是有个疑问：meanflow计算u的公式是：

```
u(zt,r,t)=v(zt,t)-(t-r)dudt
```

但是这里的瞬时速度v应该是怎么得到的呢   从remeanflow的论文里看是cfg之前用x-n直接得到解析的速度，cfg时用n-x、model输出uncond和cond按一定比例融合，（但其实没懂cfg怎么能这么融合），我就想说能不能前面用n-x，后面用n-x和model(goal_mask)，且这里的model就是刚训练好的noise_pred网络，只是后面会被detach掉

此外我融入了remeanflow里面生成t和r的分布

但是有一点不确定的地方是不知道jvp那里的参数那样给对不对，因为理论上第一个参数应该是被求导函数，应该给u才对，我给了v，但是remeanflow的代码看起来就是把原来reflow的model(理论上是v？？)直接给了，所以我看不懂model_wrapper到底是u还是v

但是为什么meanflow在sample的时候也是用vtheta去积分而不是直接用平均速度utheta？？
也许改完meanflow之后navigate里面sample的方式不用改太多，因为都是用原来的noisepred这个网络，可能可以直接减少nfes试一下

预期它最后起到的作用（u）是让流的预测方向更直，但u和v的物理意义都应该是一样的，流的积分得到的T时间步的速度值，但是publish之前已经先做了一次np.cumsum，从增量变成了轨迹waypoints了，即naction的量变成了从当前位置起的相对位置 的轨迹

但是关于一次rectify之后丢进mean flow的数据怎么处理不是很确定，也许是得到时用flownav生成一遍对应每张图片的actions，然后做好对齐之后将其作为labels输进train里面。除了meanflow之外其他block可以冻结也可以一起微调？

evaluate.py   ai说把原来的损失改成了mse_loss(n - u, naction)  （虽然ai极力争辩是n-u  但我感觉u-n比较合理）但是其实我觉得不用改比较对一点   因为eval用的ema模型其实是一样的模型只是更新比较慢而已，也就是按照它的输出，应该仍然是瞬时速度才对，除非使用了公式从v得到u，不然就应该使用一样的评估方法，并且到最后navigate.py里面输出的v其实也是一样的。但是我真的很奇怪为什么meanflow里的sample也没有转换成u

btw关于到底是t=0还是t=1的时候是噪声，我觉得我的写法应该是跟flownav保持一致而跟remeanflow反过来的，就是通过这两句

```
xt=x1 * t.view(-1, 1, 1) + noise * (1 - t).view(-1, 1, 1)  # 根据采样的时间 t 插值生成噪声点 xt
ut=(x1 - xt) / (1 - t).view(-1, 1, 1)  # 计算 xt 处的真实速度场 ut（从 xt 指向 x1 的向量）
```

并且action也是x1-x0,方向应该没问题？
