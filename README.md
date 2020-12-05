# Deep Exploration via Bootstrapped DQN
This is pytorch implmentation project of [Bootsrapped DQN](https://arxiv.org/abs/1602.04621)

## Overview
![](/img/overview.png)

Bootsrapped DQN is differ from DQN(Deep Q Network) with 4 main architecture

[1] Adapt multiple head in DQN architecture as ensemble model
[2] Add Bootsrapped Replay Memory with Bernoulli fuction
[3] Choose one head of ensemble DQN for each episod to make it run in training period
[4] Vote with best action of each heads when it comes to make action in evaluation period

## Training Code
```python

```
## Reference
