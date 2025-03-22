# Markov Chain Simulator
A scientific computing project that utilizes stochastic matrices in Markov chains to simulate and analyze probabilistic state transitions. It provides a comprehensive toolkit for generating matrices, simulating chains, and visualizing key metrics like steady-state distributions and convergence.

## Features
- **Simulate Markov Chain**: Simulates state transitions over time based on a given stochastic matrix.
- **Generate Random Stochastic Matrix**: Creates random transition matrices, ensuring valid probability distributions where each row sums to 1.
- **Plotting Functions**: Visualizes matrices, state transitions, and multiple chains using plots for a clearer understanding of transitions, steady-state behavior, and convergence patterns.
- **Calculate Steady-State Distribution**: Determines the long-term distribution where the system stabilizes, representing the equilibrium probabilities.
- **Analyze Absorption Probabilities**: Calculates the probabilities of being absorbed into certain states, crucial for systems with absorbing states.
- **Simulate Multiple Chains**: Executes multiple independent Markov chain simulations to observe the behavior from different starting points and compare results.
- **Analyze Convergence to Steady-State**: Monitors how quickly the system approaches its steady-state, offering insights into the stability and time evolution of the system.
- **Compute Mean First Passage Times**: Computes the average steps required to reach a particular state from another, useful for analyzing the efficiency of state transitions.

## Requirements
- Python 3.8+
- `numpy` for numerical computations
- `matplotlib` and `seaborn` for plotting

## Installation
1. Clone the repository: `git clone https://github.com/yourusername/markov-chain-simulator.git`
2. Navigate into the project directory: `cd markov-chain-simulator`
3. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Run the application: `python main.py`
2. Follow the menu prompts to interact with the Markov chain simulator.

## Contributing
Contributions are welcome! Please submit a pull request with your changes.

## License
[MIT License](https://opensource.org/licenses/MIT)
