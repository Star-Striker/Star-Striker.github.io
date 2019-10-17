---
layout: post
categories: posts
title: Fisrt sklearn
tags: [technology,python]
date-string: OCTOBER 17, 2019
---

用sklearn实现了简单的二元分类。

| 年龄（岁） | 年收入（万） | 是否买车 |
| ---------- | ------------ | :------- |
| 20         | 3            | 0        |
| 23         | 7            | 1        |
| 31         | 10           | 1        |
| 42         | 13           | 1        |
| 50         | 7            | 0        |
| 60         | 5            | 0        |

预测一位28岁，年收入8万的人是否买车。

```python
from sklearn import linear_model
X = [[20,3],
    [23,7],
    [31,10],
    [42,13],
    [50,7],
    [60,5]]
y = [0,
    1,
    1,
    1,
    0,
    0]
lr = linear_model.LogisticRegression()
lr.fit(X,y)

testX = [[28,8]]

label=lr.predict(testX)
print("predicted label = ",label)
prob = lr.predict_proba(testX)
print("probability = ",prob)
```

运行结果：

```
predicted label =  [1]
probability =  [[0.14694811 0.85305189]]
```

