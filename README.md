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
