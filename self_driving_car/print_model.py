from ai import Network

# Instantiate model with sample dimensions (5 input features, 3 possible actions)
model = Network(input_size=5, nb_action=3)

# Print model architecture using __repr__ method
print('\nModel Architecture:')
print(model)