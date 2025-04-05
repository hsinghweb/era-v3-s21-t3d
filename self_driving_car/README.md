# 🚗 AI-Powered Self-Driving Car Simulation 🏎️  

## 📌 Technical Overview  
**State Space**: 5-dimensional input (3 sensors + orientation + velocity)  
**Action Space**: 3 actions (straight, left, right)  
**Network Architecture**:  
```
Input(5) → FC128 → ReLU → FC128 → ReLU → Output(3)
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

## 🧠 DQN Implementation  
**Key Components**:  
- Experience Replay (capacity=100,000)  
- ε-Greedy Exploration (ε=0.1)  
- Target Network Update Frequency: Every 1000 steps  

**Hyperparameters**:  
```
Learning Rate: 0.001  
Discount Factor (γ): 0.99  
Batch Size: 500  
Replay Memory: 100,000  
Target Update: 1000 steps
```

## 🏎️ Reward System  
- **Base Reward**: +1 per frame survived  
- **Penalties**:  
  - Collision: -50  
  - Off-track: -20  
  - Sharp turns: -5  
- **Progressive Bonus**:  
  - Maintain speed > 0.8: +2/frame  
  - Center lane position: +1/frame

## 📈 Training Protocol  
1. Initial Exploration: 5000 random actions  
2. Gradual Policy Adoption:  
   - Start ε=1.0 (full exploration)  
   - Linearly decay to ε=0.1 over 50k steps  
3. Target Network Updates:  
   - Hard update every 1000 steps  

## 📂 Project Structure  
```
self_driving_car/
├── ai.py            # DQN agent implementation
├── map.py           # Simulation environment
├── car.kv           # Kivy UI configuration
├── images/          # Asset storage
└── README.md        # Project documentation
```

