# POMO-with-supervised-signal

## Description
This repository is a modified version of the original **POMO** framework. The primary goal of this project is to integrate **supervisory signals** into the training process, thereby enhancing both the efficiency and the stability of the model training.

The core codebase is built upon the repository: [[在此处粘贴原作者的仓库链接]](https://github.com/yd-kwon/POMO).

## Key Modifications

### 1. Supervised Signal Integration via Random Masking
We implemented a mechanism that uses **random masks** to select a subset of optimal paths from a supervised learning dataset. By combining this with the original Reinforcement Learning (RL) approach, we established a supervised signal adapted to the **global gradient landscape** to assist RL. 
* **TSP Results:** Significantly accelerated training speed.
* **CVRP Results:** Remarkably improved training stability.

### 2. Extended Problem Support
The original POMO framework has been or will be extended to support a broader range of Vehicle Routing Problems:
* **CVRP** (Capacitated VRP)
* **VRPTW** (VRP with Time Windows)

### 3. Framework Adaptation
The original RL-focused POMO architecture has been restructured into a hybrid framework that is fully compatible with **Supervised Learning** pipelines.

## License
This project is licensed under the MIT License.
