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

theta_0=lr.intercept_
theta_1=lr.coef_[0][0]
theta_2=lr.coef_[0][1]
print("theta_0 = ",theta_0)
print("theta_1 = ",theta_1)
print("theta_2 = ",theta_2)

testX=[[28,8]]
ratio=prob[0][1]/prob[0][0]

testX=[[28,9]]
prob_new=lr.predict_proba(testX)
ratio_new=prob[0][1]/prob[0][0]

ratio_of_ratio=ratio_new/ratio
print("ratio = ",ratio_of_ratio)

import math
theta_2_e=math.exp(theta_2)
print("theta 2 e = ",theta_2_e)
```

运行结果：

```
predicted label =  [1]
probability =  [[0.14694811 0.85305189]]
theta_0 =  [-0.04131838]
theta_1 =  -0.1973000136829152
theta_2 =  0.915557452347983
ratio =  1.0
theta 2 e =  2.4981674731438948
```

