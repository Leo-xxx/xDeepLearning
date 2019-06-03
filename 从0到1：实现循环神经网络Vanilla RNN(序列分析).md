## 从0到1：实现循环神经网络Vanilla RNN(序列分析)

原创： 技术文章 [SIGAI](javascript:void(0);) *3天前*

关注“SIGAI”，选择“星标”或“置顶”

原创技术文章，第一时间获取

转载请在文首注明：本文转自微信公众号SIGAI

全文PDF见：

http://www.tensorinfinity.com/paper_165.html

小编推荐：

六期飞跃计划还剩3个名额，联系小编，获取你的[专属算法工程师学习计划](http://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247489453&idx=2&sn=7173ea1bc50d4fc2f0599097b53a84b4&chksm=fdb68a3acac1032c455e530aba7ac9b36a4076287d2d92cd92b489ef910ca26ea3768943bc8e&scene=21#wechat_redirect)（联系小编SIGAI_NO1）

SIGAI特约作者



无双谱JX

金融架构专家

研究方向：机器学习的特定场景应用，复杂项目群管理



“In the model network...Memories are retained as stable entities and can be correctly recalled ... time ordering of memories can also be encoded...”

-- John Joseph Hopfield @Caltech , 1982

这个模型网络里...记忆是可以存取的实体，...记忆的时间线也能被编码...

导言









循环神经网络RNN，是用于序列数据分析的模型；应用场景广泛：







图像描述(image caption)；

语音识别与机器翻译；

以特定艺术风格写诗、作曲；

拟合远期资产价格曲线，试算折现盈亏；

根据社交媒体数据的情感特征，分析市场情绪和大众预期...







RNN是深度学习算法的核心构件，为了更好的理解算法，我们从动机、结构，到反向传播和学习策略，逐步分析，然后不借助深度学习框架，实现RNN模型，再应用于时序数据的分析预测，验证这个模型。

动机









之前介绍的全连接神经网络 和CNN卷积神经网络 ，其目标数据的样本，不分先后；真实世界中，样本之间可能有序列关系：排列的相对位置，或者时序关系。这类场景下，模型需要参考前序信息，来支持当前决策。









比如：

周六早上难得好天气, 正好可以（去打球 / 睡懒觉）。







结合上文，既然“天气不错”，那么“去打球”，应该高于“睡懒觉”的可能性，反之亦然。



为了获取这一类序列特征，John Joseph Hopfield1982年提出了支持记忆机制的霍普菲尔德网络(Hopfield Network)。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTQ6h7KIibzOSKmCcM3pJDfhEcdoDzFYnUETy428icqiabSaHB2iaL6FgGQw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



以此为基础，具备记忆机制的循环网络模型，逐渐演进到今天，成为更有效的RNN模型。

RNN模型结构









单层RNN

回到上面的例句，整个句子，看作多个语素排成的序列：



[周,六,早,上,难,得,好,天,气,\,, 正,好,可,以,...]



每个语素，看作时间序列[x1,x2...*x**t*]中的单个样本，RNN模型顺序接受样本输入，得到预测输出y:

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTvaybfFkNd5rDvLVlDye8OicEIx1kfu6wceqfYqULJujpiaZmbltP5Ulw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

结构中的ht，是时刻t的隐状态节点，*W**hx*,*W**hh*,*W**hy*分别是纵向和横向传播的权值参数。

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTF7fQJ6gcCnu59aLdoXyydeSFMzdV2JKZtNmXpSjFC3mb9mpZoloJEA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

模型结构中f是激活函数。



上面的RNN模型结构，也可以按照时间维度，展开隐藏节点，用下图等价表示:

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeT9ibYBfLFhlrIW6Vic69nrdhO4DQrHzY6ckNrCvrH2xbicaAEvfPChrpicw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

和单节点的等价表示一样，每个方向上的权值参数 W 是共享的。



双向RNN

有时，也会遇到由下文，反推上文的场景：



周六早上难得好天气, 正好可以（去打球 / 睡懒觉），可别叫醒我。



综合上下文，“睡懒觉”的可能性大幅提高。



这个场景下，可以通过双向RNN模型，加入逆向推理机制：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTxWXy04iaZV6OzVAvCMwbOxX6BWth1Rdw4MzG5jA72uS9YyabOEibK0RQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

同正序列一样，逆序列的权参W'，在每个方向上也是共享的。



整个模型的参数量、隐状态节点数，较单向模型翻了一倍，模型也能表达更深层的语义信息。



多层RNN

遇到更复杂的序列信息呢？



回顾CNN卷积神经网络模型，为了获得更丰富的特征，可以调整卷积核或者Conv-Pool单元的数量。



与之对应，RNN的多层模型，通过叠加隐藏层，来抽取更深层的序列特征：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeT1aEzvuoXHsWmNBbrWoMbicmIh6h9d5rPWica8Bds3mHCYYJPuZN0qYMQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在多层RNN结构中，每个隐藏层，各个方向上共享权值参数W。

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTKE1HOfhCaeqbX619B3Jrt8lI4z0ciaRzJjThbLVSjamG6d9GFpBE9Ng/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

为方便反向传播推导，上面的计算，可以写为更一般的前向传播算法表达式。

前向传播算法









RNN模型的输入X是 D 维向量，W 是权参矩阵，任意时刻 t ，隐藏节点的原始输出，是该时刻输入Xt，和 t-1 时刻隐层节点输出的加权和：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTmYQvZApLaW8XLia20LbzkjpvViaCvMdolqtfHA1ibFdOjGLibZcnyfVdlg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如果选用双曲正切函数 tanh，作为节点输出的激活函数，则隐藏节点激活后输出为：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

对RNN多层模型，第l层 t 时刻隐藏节点，原始输出成为：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

整个RNN单元,在最后一层L的输出为：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

RNN的算法表达式中，也可以把*W**hx*和*W**hh*拼接为一个权参矩阵W，同时拼接上一层输入和前一时刻隐藏节点输出[*x**t*,h*t-1*]，再和权参W做仿射变换：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTsyeeyKAoeXovZ9ra0AEiaEW284f4TYA6h0bicuJr55MNX8eMWrxpXUJQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

训练和预测时，只要表达式保持一致，两种方法的结果是等价的。这里采用权参分开来书写表达式，是为了便于观察反向传播算法的推导步骤。



以上是Vanilla RNN前向计算的全部表达式。



对于反向传播，主流的深度学习框架，通过自动微分实现了支持，为什么我们还要了解反向传播的原理呢？



首要原因是，反向传播会有抽象泄露(Leaky Abstractions)问题。

抽象泄露















All non-trivial abstractions, to some degree, are leaky.
--Joel Spolsky，CEO of Stack Overflow，2002

所有的复杂抽象，或多或少，都有泄露。









抽象泄露法则告诉我们：所有试图隐藏复杂细节的抽象，都不能做到完全封装，总会有底层机制泄露到抽象之上，给使用者带来麻烦；表层抽象，初期节约了使用成本，然而长远看，这些漏洞会以各种“灵异缺陷“，或者性能问题的形式，呈现出来，要解决这些漏洞，则需要深入了解抽象之下的原理。



Joel Spolsky用好莱坞速运的例子，形象的类比了TCP协议对IP协议的抽象泄露。









Imagine that we had a way of sending actors from Broadway to Hollywood that involved putting them in cars and driving them across the country. Some of these cars crashed, killing the poor actors. Sometimes the actors got drunk on the way and shaved their heads or got nasal tattoos, thus becoming too ugly to work in Hollywood, and frequently the actors arrived in a different order than they had set out, because they all took different routes. Now imagine a new service called Hollywood Express, which delivered actors to Hollywood, guaranteeing that they would (a) arrive (b) in order (c) in perfect condition. The magic part is that Hollywood Express doesn’t have any method of delivering the actors, other than the unreliable method of putting them in cars and driving them across the country. Hollywood Express works by checking that each actor arrives in perfect condition, and, if he doesn’t, calling up the home office and requesting that the actor’s identical twin be sent instead. If the actors arrive in the wrong order Hollywood Express rearranges them. If a large UFO on its way to Area 51 crashes on the highway in Nevada, rendering it impassable, all the actors that went that way are rerouted via Arizona and Hollywood Express doesn’t even tell the movie directors in California what happened. To them, it just looks like the actors are arriving a little bit more slowly than usual, and they never even hear about the UFO crash.








假设我们要把演员从百老汇送到好莱坞，方法是用汽车载着演员横穿大陆。演员们可能路遇车祸，不幸去世；也可能演员在途中喝醉酒，跑去剃了光头，或是纹了纳粹刺青，总之丑到无法工作；此外，演员们常常因为各走各路而先发后至。现在我们有了一项新服务：好莱坞速运，能确保演员们以（a）良好状态（b）依次（c）到达。秘诀是不再去关注、应对旅途上各种不可靠状况，而是在终点确认每位演员按时到达，状态良好，否则就呼叫总部，送来演员的双胞胎。如果演员没有依次到达，就先排好队再露面。这样，就算有架51区大飞碟，坠毁在内华达的主干道上，阻塞了交通，让所有在途演员都要改道亚利桑那州，加州的导演也永远也听不到什么飞碟坠毁事件，在导演看来，演员们只是比平常晚到了一会儿。



同样的，TCP协议号称可靠，然而并不是，因为所依赖的IP层及以下协议，原本就不是100%可靠的。首先，物理线路不通，任何IP包都不能通过，TCP也不会工作；更常见的情况下，交换机超负荷，IP包仅能部分通过，TCP看上去还在工作，然而慢得不得了。



TCP试图对不可靠的底层提供完整的抽象，却不能真的保护你免受底层问题的影响。



在RNN这里，一个是典型的抽象泄露场景，是梯度传递问题，为了理解它，需要深入反向传播的算法原理。

反向传播算法









RNN的反向传播，与CNN相比，多了一条按照时间反向传导的通道，也被称为**BPTT**（back-propagation through time）算法，下面分解来看。



误差的反向传播

由于任意时刻t，第l层隐藏节点的输出：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTqLa64v9JytrZNrozAcj7j4RJtYQhzNU4DibaCaMZyILpiauiaccDiccXpw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

m反向传递到隐藏节点原始输出这个环节，其误差 E，来自层级和时间两个方向，可由复合求导法则，推得这个误差：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTKKHdBd5icVdNVGmXfMByj6Agbh8UUiadgRcVGatHkAoWOGBKm6j3lT5g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

(式1）中diag(f')表示由向量f'构成的对角阵，⊙是CNN卷积神经网络 中介绍过的Hadamard乘积运算。



观查（式1）第二项，含有双曲正切函数的导函数。



激活函数的导函数

由原函数：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTppd8lQUdh7nicPCibgHL4uQbqR4SeRibeyLialMeM08HOWUT8X1wkn4cibg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

可推得激活函数的导函数：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTquROHoCCkqib0Bpq0SyAG43xzcDNiciaRo1er1Cmr2KTMe3eBXPMOrtTg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

所以：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTnd4jQOAIA32eicOicicuXzr1NPcGjNibBqHV4LXaR918wCsORPpJRYVNNw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这个性质好在：前向传播时，保留原始激活输出的数值结果，反向传播计算时，可以直接复用。



有了隐藏节点的误差，容易推导沿时间和层次继续传递的误差项：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTOq2wzuR85u8Qc9CH4BOgiaRjXhv5noo7Cm4G5evgghmI52lRuJey9YA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



权参梯度计算

在时间方向上，权参*W**hh*在t 时刻节点的梯度，参考全连接神经网络的反向传播，有：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

同样推出在层次方向上，权参在 t 时刻节点的梯度：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

以及偏置项bias的在 t 时刻节点的梯度：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

最后，把同层各个时刻节点梯度加总，得到多层RNN模型中，任意一层权参和偏置项的梯度：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTkYSrMsdUBLmgMUbSWicGYB0ddicHxj0qksleRtNsQafJZcsHicUp5zPXQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

至此，完成了多层RNN模型中，BPTT反向传播算法的全部误差传递，和参数梯度计算。



RNN模型的基本原理，并不复杂。然而，要根据数据场景，适当的完成模型训练，却不容易，BPTT算法是基于梯度传递(gradient based)的算法，模型训练会面临梯度传递问题。

RNN的梯度传递问题









RNN模型在同一层上共享权值参数，依时间步循环展开迭代计算；观察反向传播，可以看到沿时间步反向传递的梯度，在所有的隐藏节点上，总是和同一个矩阵做乘积运算(multiply)。



如果用一个被乘数a，去不断累乘另外一个乘数b，最终的乘积要么趋近于0（|b|<1）,要么膨胀到无穷大（|b|>1）。



回顾RNN误差沿时间步的反向传播：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTdSGTeic8xLACr5yqkXoedgWvGY24fKcOiapeIlmrjeXGxAtNibzxs0fgg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

对权参*W**hh*，把乘数b，换成矩阵的最大本征值，会发生类似的情况。



梯度趋近于0和无穷大的情况，分别被称为梯度弥散( vanishing gradient)和梯度爆炸( exploding gradient)。



梯度弥散问题，可以通过改进优化方法（Rmsprop/Adam）或者下篇介绍的RNN变体模型来缓解；



梯度爆炸问题，可以通过牺牲训练效率，调低学习率来改善；这里介绍另一种应对方法：梯度裁剪。

梯度裁剪方法









梯度裁剪(gradient clipping)，是一种简单有效的方法。



这种方法，在梯度突然变大的情况下，将它收缩(rescale)到特定阈值（threshold）之下：



观察梯度变化的方式是跟踪它的L2范数：

当![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)时

更新梯度值为：

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

工程实践中，用当前时间步下，所有权参、偏置的L2范数的平方和再开方，得到一个标量，再和阈值比较大小，并确定梯度的收缩系数，简洁高效地实现计算。



新引入的一个超参数threshold，需要根据数据场景来设置和调整，实验表明Razvan et al.(2012)，模型对这个阈值参数不大敏感，即使取值很小，算法也有不错结果。



梯度裁剪方法，也可以看作是根据梯度的范数，自适应的调整学习率，但是和自适用优化方法不同；自适应优化方法，以梯度的累计统计量来更新梯度，起到加快模型收敛的效果；梯度裁剪的效果不是加速收敛，而是通过直接调整每个时间步的梯度，来缓解梯度爆炸问题。

实现









基于上述推导，可以不借助深度学习框架，实现RNN模型的前向和反向传播算法。



前向传播在时间节点和层级，两个方向上分别计算：

```

def rnn_step_forward(self, x, prev_h, Wx, Wh, b):
"""
Inputs:
   - x: Input data for this timestep, of shape (N, D).
   - prev_h: Hidden state from previous timestep, of shape (N, H)
   - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
   - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
   - b: Biases of shape (H,)
Returns a tuple of:
   - next_h: Next hidden state, of shape (N, H)
   - cache: Tuple of values needed for the backward pass.
"""
next_h, cache = None, None
z = np.matmul(x,Wx)+np.matmul(prev_h,Wh) +b
next_h = np.tanh(z)
dtanh = 1. - next_h * next_h
cache=(x, prev_h, Wx, Wh, dtanh)
return next_h, cache
```

反向传播，同样在两个方向上，计算误差传递和权参梯度：

```
def rnn_step_backward(self, dnext_h, cache):
"""
Inputs:
   - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
   - cache: Cache object from the forward pass
Returns a tuple of:
   - dx: Gradients of input data, of shape (N, D)
   - dprev_h: Gradients of previous hidden state, of shape (N, H)
   - dWx: Gradients of input-to-hidden weights, of shape (D, H)
   - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
   - db: Gradients of bias vector, of shape (H,)
"""
dx, dprev_h, dWx, dWh, db = None, None, None, None, None
x, prev_h, Wx, Wh, dtanh = cache
dz = dnext_h * dtanh
dx = np.matmul(dz,Wx.T)
dprev_h = np.matmul(dz,Wh.T)
dWx = np.matmul(x.T,dz)
dWh = np.matmul(prev_h.T,dz)
db = np.sum(dz,axis=0)
return dx, dprev_h, dWx, dWh, db

```

反向传播，同样在两个方向上，计算误差传递和权参梯度：



不算注释的话，以上17行源码，实现了RNN模型的核心算法。

预测序列数据









为了验证上述算法，可以用一组序列数据，作为目标问题，来观察模型预测损失。



用正、余弦函数叠加，生成训练和验证数据：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTHuPiavT1ibCiaQGjukfqlzhQaeyYfGuvAGWibRphJxneASw5D4v6JibObWQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

序列数据的图像是这样的：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTUuiarBd8m6nJDDxZpAFTwibYwd3SqdSH6HHZJZrqxhIJxia0gichxSicpbA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



给数据增加一些随机噪声，图像成为：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTP7EDQQLcGMIszTVWDJvE59NhRzvFps98kfDz6Vbsbooxd47bl9lLDg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



对函数输出，采样生成离散数据集。



如果用平方误差作为损失函数，则D维样本数据的预测损失为：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTVJtuugRyWnffImBpjC2Htk3A4RFZ8ibWW7tqQ3SfuMTku9lDmIV9icicQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在这个序列预测场景中，以一个时段的数据序列作为输入，预测下一刻的输出；在RNN最后一层，各个时间节点可以有多个输出，我们只需要获取最后一个时间节点的数值结果y=yt，来计算预测损失。



误差损失为：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTsgD9jIMlqugFdKe2nmjd0Tz1GBVMQn4wMpibMr90xukZDsUUO8TMwPw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这个误差损失，反向传递给BPTT算法，得到各个层级节点的误差和权参梯度。



我们用上面实现的算法，搭建一个三层RNN模型：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTiaSNQ24mib921m5efAyGpTmvgsKkIDPN3b99XeN0gHcESGOohNMOTLsQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

最后的节点作为模型预测输出，和真实结果![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTFbnzT4ajGG78LQgBu2Zibd7ictOnticoIwHdZl8OCzGkeicPTIWYowp0wQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)一起，计算损失 。



这个预测场景的学习策略优化算法，同样不借助深度学习框架，复用卷积神经网络(实现篇)中完成的Adagrad算法。

https://zhuanlan.zhihu.com/p/49205794

完整模型和序列预测算法的实现，可以在github上获取源码。



https://github.com/AskyJx/xDeepLearning



验证结果









模型在训练和验证数据上，都可以正常收敛。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTV1EqicVQDcQQTsqk7GUoIT3OOWmLf8hBHOONYyYE6hLRP3ICdAgetEA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

经过600轮迭代训练，模型输出趋于稳定。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTOT5EhIoHoMib3DiaVUibkYbE6MrFbQw6GzGEulG6dfUXNWdyxAomj3VCA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

预测输出曲线，几乎与真实数据重合。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/75DkJnThACnnMhl1PRx5LX0l8X2fQHeTvjCUYvbO90nXwiaicubDbztKYjTAhibXtul6H8dmPyrueAwdB7KBNpqCg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这个实现不借助深度学习框架，换用不同的激活函数和学习策略优化方法，都很方便，可以根据不同的数据特性，自由订制组合模型构件，灵活适应不同的场景。



RNN循环神经网络，引入记忆机制，得以抽取序列特征，且能通过增加隐藏节点和隐藏层级，提高模型的表达能力，开辟了深度学习算法和适用场景的新方向。然而，时序数据中，远近不同样本，对输出决策的影响权重，往往不同；此外，随着RNN模型深度增加，梯度传递问题也使模型不易训练。



下一次，我们继续了解Vanilla RNN模型的变体：长短时记忆网络LSTM，对这些问题的解决方法，然后同样不借助深度学习框架，完成前向和反向传播算法的实现。

（完）







参考：

[1] The Stanford CS class CS231n

[2] The Unreasonable Effectiveness of Recurrent Neural Networks

[3] Yes you should understand backprop

[4 ] The Law of Leaky Abstractions

[5] Bengio, Yoshua, Simard, Patrice, and Frasconi, Paolo.Learning long-term dependencies with gradient descent is difficult. Neural Networks, IEEE Transactions on, 5(2):157–166, 1994.

[6] Razvan Pascanu, Tomas Mikolov, Yoshua Bengio. Understanding the exploding gradient problem. CoRR abs/1211.5063 ,2012.

[7] J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities", Proceedings of the National Academy of Sciences of the USA, vol. 79 no. 8 pp. 2554–2558, April 1982.













本文为SIGAI原创

 如需转载，欢迎发消息到本订号







![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACkytNVK6xFicGLUBT0jHNtiaMtORW5tMsrQZvLBbgbPSggjOrdMjticicL9OuXfpIkuYd7v6760MWW20A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



全文PDF见：http://www.tensorinfinity.com/paper_165.html

看完欢迎转发哟

[阅读原文](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247490075&idx=1&sn=d36f9ab753ce93ba836e0d2348c3b998&chksm=fdb6878ccac10e9af4727b1010b8d2667c1ae890f918772ee78ca90117b095baced54dff6412&mpshare=1&scene=1&srcid=&key=90581f21d61583cc655a4e576e4392e4d6a0e33db562b013c296c8ab74509db3782a53160d3f2f5e79b04319fe65eda94849a65eb6d9d17bd964a1e1b6960e2136715f8525d66587040f6ecd58124b19&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=%2BtDZ2Al0VM5wjz5XAzAxV1jJFwepKB91N4744YqAfvwEIleHxJyeJlLibQdxfrJN##)





微信扫一扫
关注该公众号