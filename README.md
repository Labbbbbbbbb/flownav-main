# FlowNav: Combining Flow Matching and Depth Priors for Efficient Navigation

> [!NOTE]  
> This  code was forked from [utn-air/flownav: [IROS25\] Combining Flow Matching and Depth Priors for Efficient Navigation](https://github.com/utn-air/flownav?tab=readme-ov-file)

## 

更改的部分都写在`meanflow`文件夹内，目前的预想是运行时直接在`loop.py`中将train函数从原来的`flownav\training\train.py`中定义的train改为`meanflow`中的train

即`flownav\training\loop.py`:

```
#from flownav.training.train import train
from meanflow.train import train
```

evaluate.py没有修改

ps:所有带`zyt`tag的部分都是修改过的，尤其是`deployment\src\navigation`中代码修改较多，`flownav`中的部分参数也是后面自己添加的
