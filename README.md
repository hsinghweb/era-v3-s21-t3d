# 🚗 ERA-V3 Self-Driving Car with TD3 🤖

Welcome to the ERA-V3 Self-Driving Car project! This repository implements a Twin Delayed Deep Deterministic Policy Gradient (TD3) approach for autonomous vehicle control in a simulated environment.

## 🌟 Key Features

- **Continuous Action Space**: Precise control over steering and acceleration
- **Twin Delayed Policy Updates**: Enhanced stability in learning
- **Advanced Reward System**: Optimized for smooth and efficient driving
- **Real-time Visualization**: Interactive Kivy-based simulation environment

## 🏗️ Project Structure

```
self_driving_car/
├── ai.py            # TD3 agent implementation
├── map.py           # Simulation environment
├── car.kv           # Kivy UI configuration
├── images/          # Asset storage
└── README.md        # Detailed documentation
```

## 🔍 Technical Overview

- **State Space**: 5D continuous state (3 sensors + orientation + velocity)
- **Action Space**: 2D continuous actions (steering angle + acceleration)
- **Architecture**: Actor-Critic networks with twin Q-functions

For detailed implementation and usage instructions, please check the documentation in the `self_driving_car` directory.
