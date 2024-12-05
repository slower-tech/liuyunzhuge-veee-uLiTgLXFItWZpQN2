
![](https://img2024.cnblogs.com/blog/3524016/202412/3524016-20241204091315915-1943612086.png)


上一篇：《人工智能模型学习到的知识是怎样的一种存在？》


**序言：**在接下来的几篇中，我们将学习如何利用 TensorFlow 来生成文本。需要注意的是，我们这里并不使用当前最热门的 Transformer 模型，而是探讨传统的机器学习方法。这么做的目的，是让你对当前主流的生成式人工智能模型有一个对比性的理解。通过了解这些传统模型和现代 Transformer 模型的差异，你会发现，现代生成式模型的成功，背后的技术，其实就是“规模法则”（Scaling Law）的推动。你可能会好奇，为什么以前的模型无法以同样的方式取得成功？这主要归结为两个原因：


**1\. 硬件计算能力的不足：**早期的硬件资源和计算能力无法支撑大规模模型的训练，尤其是在数据量和模型规模不断扩展时，计算能力成了瓶颈。


**2\. 传统模型的并行化困难：**像 RNN、CNN、DNN 等模型，虽然在特定任务上有优势，但它们在处理长篇幅或复杂依赖关系的文本时存在局限，尤其是 RNN 和 LSTM 的训练效率低，无法像 Transformer 那样进行高效的并行计算。


而 Transformer 的核心优势正是在于它能够充分并行化计算，利用自注意力机制高效捕捉长距离依赖，这也是它取得成功的关键原因。因此，随着硬件和计算能力的提升，Transformer 终于能够脱颖而出，成为了现代生成式模型的主流架构。


**让我们开始吧！**


You know nothing, Jon Snow


the place where he’s stationed


be it Cork or in the blue bird’s son


sailed out to summer


old sweet long and gladness rings


so i’ll wait for the wild colleen dying


你一无所知，乔恩·雪诺


他驻扎的地方


无论是在科克，还是在蓝鸟的儿子


航行到夏天


古老的甜美、长久和欢乐的钟声响起


所以我会等待那个野性姑娘的死去


这段文字是由一个非常简单的模型生成的，训练的语料库很小。


我稍微进行了改进，添加了换行符和标点符号，但除了第一行外，其余的内容都是由你将在本篇中学习构建的模型生成的。


它提到一个“野性姑娘的死去”挺有意思——如果你看过乔恩·雪诺出自的那个剧集，你会明白为什么！


在过去几章中，你看到如何使用TensorFlow处理基于文本的数据，


首先将其标记化为数字和序列，这些可以被神经网络处理，


然后使用嵌入向量来模拟情感，最后利用深度和循环神经网络来分类文本。


我们使用了一个小而简单的讽刺数据集，来说明这一切是如何运作的。


在这一篇中，我们将换个方向：不再分类现有文本，而是创建一个神经网络，


它可以预测文本。给定一个文本语料库，它将尝试理解其中的词汇模式，以便当给定一个新的文本片段（即种子）时，预测接下来应该出现哪个词。


一旦预测出一个词，种子和预测的词就成为新的种子，接下来的词可以继续预测。


因此，当神经网络在一个文本语料库上训练时，它可以尝试以类似的风格创作新的文本。


为了创作上面的这段诗歌，我收集了一些传统爱尔兰歌曲的歌词，


用它们训练了一个神经网络，并用它来预测词汇。


我们将从简单的开始，用少量文本来说明如何建立一个预测模型，


最后创建一个包含更多文本的完整模型。之后，你可以尝试看看它能创作出什么样的诗歌！


开始时，你必须以不同于之前的方式处理文本。


在前几篇中，你将句子转化为序列，然后基于其中标记的嵌入向量进行分类。


而在创建可以用于训练预测模型的数据时，有一个额外的步骤，


即需要将序列转化为输入序列和标签，其中输入序列是一组词汇，标签是句子中的下一个词。


然后，你可以训练一个模型，使其将输入序列与其标签匹配，以便在未来的预测中，选择一个接近输入序列的标签。


**将序列转换为输入序列**


在预测文本时，你需要用一个包含相关标签的输入序列（特征）来训练神经网络。将序列与标签匹配是预测文本的关键。


举个例子，如果你的语料库中有一句话“Today has a beautiful blue sky”，你可以将其拆分为“Today has a beautiful blue”作为特征，和“sky”作为标签。然后，如果你输入“Today has a beautiful blue”进行预测，模型很可能会预测出“sky”。如果在训练数据中你还有一句话“Yesterday had a beautiful blue sky”，按同样的方式拆分，而你输入“Tomorrow will have a beautiful blue”进行预测，那么接下来的词很有可能也是“sky”。


通过大量的句子，训练一组词语序列，其中下一个词就是标签，你可以迅速构建出一个预测模型，使得从现有的文本中能够预测出句子中最可能出现的下一个词。


我们从一个非常小的文本语料库开始——一个来自1860年代的传统爱尔兰歌曲的片段，其中部分歌词如下：


In the town of Athy one Jeremy Lanigan


Battered away til he hadnt a pound.


His father died and made him a man again


Left him a farm and ten acres of ground.


He gave a grand party for friends and relations


Who didnt forget him when come to the wall,


And if youll but listen Ill make your eyes glisten


Of the rows and the ructions of Lanigan’s Ball.


Myself to be sure got free invitation,


For all the nice girls and boys I might ask,


And just in a minute both friends and relations


Were dancing round merry as bees round a cask.


Judy ODaly, that nice little milliner,


She tipped me a wink for to give her a call,


And I soon arrived with Peggy McGilligan


Just in time for Lanigans Ball.


在阿西镇，有个杰里米·拉尼根


他拼命干活，直到一分钱都没剩


他的父亲去世后又让他变成了男人


给了他一块农田和十英亩的土地


他为朋友和亲戚们举办了一个盛大的派对


当他面临困境时，他们没有忘记他


如果你肯听，我会让你的眼睛闪闪发光


讲讲拉尼根舞会上的争吵和混乱


我自己当然也收到了免费邀请


因为我可以请任何我喜欢的女孩和男孩


很快，朋友和亲戚们


就像蜜蜂围着桶一样开心地跳起舞来


朱迪·奥达利，那位可爱的帽子商


她对我眨了眨眼，示意我给她打个招呼


我很快就和佩吉·麦吉利根一起到达


刚好赶上了拉尼根舞会


将这段文字合并成一个字符串，并用 \\n 作为换行符。然后，语料库就可以像这样方便地加载和标记化：


tokenizer \= Tokenizer()


data\="In the town of Athy one Jeremy Lanigan \\n Battered away ... ..."


corpus \= data.lower().split("\\n")


tokenizer.fit\_on\_texts(corpus)


total\_words \= len(tokenizer.word\_index) \+ 1


这个过程的结果是将单词替换成它们的标记值，如图8\-1所示。


![](https://img2024.cnblogs.com/blog/3524016/202412/3524016-20241204091423173-671084326.png)



```
                                                    图8-1. 对句子进行标记化

```

为了训练一个预测模型，我们应该进一步处理——将句子拆分成多个较小的序列。例如，我们可以得到一个由前两个标记组成的序列，再一个由前三个标记组成的序列，依此类推（见图8\-2）。


![](https://img2024.cnblogs.com/blog/3524016/202412/3524016-20241204091644029-772117780.png)



```
                                                    图8-2. 将一个序列转化为多个输入序列

```

为了做到这一点，你需要遍历语料库中的每一行，并使用texts\_to\_sequences将其转化为标记列表。然后，你可以通过循环遍历每个标记，制作出一个包含所有标记的子列表。


这里是代码：


input\_sequences \= \[]


for line in corpus:


token\_list \= tokenizer.texts\_to\_sequences(\[line])\[0]


for i in range(1, len(token\_list)):


n\_gram\_sequence \= token\_list\[:i\+1]


input\_sequences.append(n\_gram\_sequence)


print(input\_sequences\[:5])


一旦你得到了这些输入序列，你就可以对它们进行填充，使它们具有统一的形状。我们将使用前填充（见图8\-3）。


![](https://img2024.cnblogs.com/blog/3524016/202412/3524016-20241204091539310-761331700.png)



```
                                  图8-3. 对输入序列进行填充

```

为了做到这一点，你需要找到输入序列中最长的句子，然后将所有序列填充到那个长度。这里是代码：


max\_sequence\_len \= max(\[len(x) for x in input\_sequences])


input\_sequences \= np.array(pad\_sequences(input\_sequences,


maxlen\=max\_sequence\_len, padding\='pre'))


最后，一旦你得到了填充后的输入序列，就可以将它们分成特征和标签，其中标签就是输入序列中的最后一个标记（见图8\-4）。


![](https://img2024.cnblogs.com/blog/3524016/202412/3524016-20241204091604881-669306459.png)



```
                                图8-4. 将填充后的序列转化为特征（x）和标签（y）

```

在训练神经网络时，你将每个特征与其对应的标签进行匹配。举个例子，像这样的输入序列 \[0 0 0 0 4 2 66 8 67 68 69]，它的标签就是 \[70]。


这里是将标签与输入序列分开的代码：


xs, labels \= input\_sequences\[:, :\-1], input\_sequences\[:, \-1]


接下来，你需要对标签进行编码。现在它们只是标记，比如图8\-4顶部的数字2。但是，如果你想把标记作为分类器的标签使用，它就必须映射到一个输出神经元上。因此，如果你想要分类n个词，每个词都是一个类别，你就需要n个神经元。这时，控制词汇表的大小非常重要，因为词汇越多，你需要的类别（神经元）就越多。记得在第2章和第3章中，你用Fashion MNIST数据集来分类服饰项目，当时你有10种不同的服饰类型吗？那时你需要在输出层有10个神经元。如果这次你想预测最多10,000个词汇呢？你就需要一个包含10,000个神经元的输出层！


另外，你还需要对标签进行一热编码，以便它们能匹配神经网络所需的输出。看一下图8\-4。如果神经网络输入的X是一个由一系列0后跟4组成的序列，你希望预测结果是2，而网络实现这个预测的方法是通过让输出层包含词汇大小数量的神经元，其中第二个神经元的概率最大。


为了将标签编码成一组Y，然后用来训练，你可以使用tf.keras中的to\_categorical工具：


ys \= tf.keras.utils.to\_categorical(labels, num\_classes\=total\_words)


你可以在图8\-5中看到这个过程的可视化效果。


![](https://img2024.cnblogs.com/blog/3524016/202412/3524016-20241204091714076-1645368476.png)



```
                                                  图8-5. 标签的one-hot编码

```

这是一个非常稀疏的表示方式，如果你有大量的训练数据和潜在的词汇，内存会被很快消耗掉！假设你有100,000个训练句子，词汇表中有10,000个单词——你需要10亿字节的内存来存储这些标签！但这是我们必须这样设计网络的原因，因为我们要进行词汇的分类和预测。


总结：本章通过一个简单的例子介绍了如何用 TensorFlow 训练一个文本生成模型，重点在于如何通过神经网络学习文本中的词汇模式，从而生成与输入文本相似的内容。通过理解数据处理、模型训练及其背后的原理，我们为进一步深入学习生成式人工智能奠定了基础，在下一篇中我们将再次用一完整的例子跟大家一起学习如何动手制作一个AI模型，并训练它来预测（生成）下一个单词。


 本博客参考[楚门加速器](https://shexiangshi.org)。转载请注明出处！
