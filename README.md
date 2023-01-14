# UAV-assisted-MEC


> **Basic Introduction** <br><br>
The arrival of 5G has paced the research towards edge computing so that computation can be achieved in optimal time in a close proximity. There are Fixed location Edge Computing Data Centers, but they have many disadvantages. Since `UAVs(Unmanned Aerial Vehicles)` can be deployed on short notice, they can play an instrumental role in wireless networks by increasing coverage and enhancing communication quality. A lot of research is in progress to achieve optimal `MEC(Multiple Access Edge  Computing)` service which is mobile. This project is a step towards trying to create a Reinforcement Learning Model to make use of UAVs (because of their high mobility and flexible deployment) to automate the scheduling for `UEs (User Equipments)` and achieve optimal offloading policy with the target to achieve minimum average computation latency among all the UEs.

<br>

- This repository is an implementation of the concepts and techniques outlined in the [paper](https://link.springer.com/article/10.1007/s11276-021-02632-z). It builds upon the ideas presented in the paper, with modifications to both the algorithm and environment to make it more practical and enhance its performance.
- It consists of implementation of UE scheduling using `DDPG(Deep Deterministic Policy Gradient)` and `SAC(Soft Actor Critic)` algorithm
- **Next Step** 
  1. Obtain comparable outcomes utilizing `A3C (Asynchronous Advantage Actor Critic)` algorithm.
  2. Moving towards `MARL(Multi-Agent Reinforcement Learning)` for all algorithms

## Requirements
```
gym
tensorflow
torch
numpy
pandas
```


## Environment
The Environment comprises N mobile UEs randomly initially scattered in a 100m X 100m 2D space. The UEs are in a constant state of motion, traveling in random direction and at the speed v_ue.

### State
```
[Battery left in UAV, location of UAV, Total Task Left Sum, Task Size of all N UEs, location of all N UEs, Block Flag for all N UEs]
```

### Action
```
[UE id, flight angle, flight speed, offloading-ratio]
```
where
  1. **UE id** - is which User Equipment(UE) is to be served for that particular time quantum 
  2. **flight angle** - the direction in which UAV should fly to serve the next ue 
  3. **flight speed** - the speed s of UAV, 0 <= s <= v_max where v_max is maximum speed the UAV can achieve
  4. **off-loading ratio** - the ratio of task to be uploaded in the UAV

### Reward
The reward function consists of a weighted sum of computations performed in a specific time quantum and the number of UEs that complete their computations inside that time quantum.

### Results

![Result 1](/results/result1.png)
![Result 2](/results/result2.png)

Models were trained for 25k episodes.
The graph shows a significant drop in the average time delay for the models.
