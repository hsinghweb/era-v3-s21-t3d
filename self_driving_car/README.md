# 🚗 AI-Powered Self-Driving Car Simulation 🏎️  

## 📌 Technical Overview  
**State Space**: 5-dimensional continuous input (3 sensors + orientation + velocity)  
**Action Space**: Continuous 2D action space for steering and acceleration  
**Network Architecture**:  
```
Actor: Input(5) → FC400 → ReLU → FC300 → ReLU → tanh → Output(2)
Critic (Twin): Input(5+2) → FC400 → ReLU → FC300 → ReLU → Output(1)
```

## 🛠️ Installation  
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch==2.0.1 kivy==2.2.1 numpy==1.24.3 matplotlib==3.7.2 pillow==10.0.0

# Run simulation
python map.py
```

## 🧠 TD3 Implementation  
**Key Components**:  
- Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Experience Replay (capacity=100,000)  
- Target Networks with Soft Updates (τ=0.005)
- Action Noise for Exploration (σ=0.1)

**Hyperparameters**:  
```
Learning Rate: 0.001
Discount Factor (γ): 0.99  
Batch Size: 500  
Replay Memory: 100,000  
Policy Update Frequency: 2
Policy Noise: 0.2
Noise Clip: 0.5
```

## 🏎️ Reward System  
- **Base Reward**: +1 per frame survived  
- **Penalties**:  
  - Collision: -50  
  - Off-track: -20  
  - Being stuck: -0.5 × stuck_counter  
- **Progressive Bonus**:  
  - Momentum bonus: +0.2 × momentum  
  - Continuous movement: Momentum increases by 0.1 (max 1.0)

## 📈 Training Protocol  
1. Initial Exploration: Gaussian noise (σ=0.1)
2. Twin Critics:
   - Minimize MSE between Q-values and targets
   - Use minimum of twin Q-values for targets
3. Delayed Policy Updates:
   - Update policy every 2 critic updates
   - Soft target network updates (τ=0.005)

## 📂 Project Structure  
```
self_driving_car/
├── ai.py            # TD3 agent implementation
├── map.py           # Simulation environment
├── car.kv           # Kivy UI configuration
├── images/          # Asset storage
└── README.md        # Project documentation
```

