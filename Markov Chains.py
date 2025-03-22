import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MarkovChain:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
        self.num_states = transition_matrix.shape[0]

    def is_stochastic(self):
        """Check if the transition matrix is stochastic."""
        row_sums = self.transition_matrix.sum(axis=1)
        return np.allclose(row_sums, 1)

    def next_state(self, current_state):
        """Return the next state given the current state."""
        return np.random.choice(self.num_states, p=self.transition_matrix[current_state])

    def simulate(self, initial_state, num_steps):
        """Simulate the Markov chain for a given number of steps."""
        states = [initial_state]
        for _ in range(num_steps):
            next_state = self.next_state(states[-1])
            states.append(next_state)
        return states

    def steady_state_distribution(self):
        """Calculate the steady-state distribution."""
        eigvals, eigvecs = np.linalg.eig(self.transition_matrix.T)
        steady_state = eigvecs[:, np.isclose(eigvals, 1)]
        steady_state = steady_state / steady_state.sum()
        return steady_state.real.flatten()

    def absorption_probabilities(self):
        """Calculate absorption probabilities if there are absorbing states."""
        absorbing_states = np.where(np.isclose(np.diag(self.transition_matrix), 1))[0]

        if len(absorbing_states) == 0:
            print("No absorbing states found.")
            return None

        Q = self.transition_matrix[np.ix_(~np.isin(range(self.num_states), absorbing_states),
                                          ~np.isin(range(self.num_states), absorbing_states))]

        R = self.transition_matrix[np.ix_(~np.isin(range(self.num_states), absorbing_states),
                                          absorbing_states)]

        I = np.eye(Q.shape[0])
        N = np.linalg.inv(I - Q)

        absorption_probs = N @ R
        return absorption_probs


def plot_markov_chain(states, num_states):
    """Plot the states of the Markov chain over time."""
    plt.figure(figsize=(10, 6))
    plt.title("Markov Chain Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("States")
    plt.scatter(range(len(states)), states, c=states, cmap='viridis', marker='o')
    plt.yticks(range(num_states))
    plt.grid()
    plt.show()


def plot_transition_matrix(transition_matrix):
    """Visualize the transition matrix using a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap='Blues', square=True)
    plt.title("Transition Matrix Heatmap")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()


def generate_random_stochastic_matrix(size):
    """Generate a random stochastic matrix of given size."""
    matrix = np.random.rand(size, size)
    return matrix / matrix.sum(axis=1, keepdims=True)


def simulate_multiple_chains(transition_matrix, initial_states, num_steps):
    """Simulate multiple Markov chains from different initial states."""
    chains = []
    for state in initial_states:
        mc = MarkovChain(transition_matrix)
        states = mc.simulate(state, num_steps)
        chains.append(states)

    return chains


def plot_multiple_chains(chains):
    """Plot multiple Markov chain simulations."""
    plt.figure(figsize=(10, 6))

    for i, states in enumerate(chains):
        plt.plot(states, label=f'Chain {i + 1}')

    plt.title("Multiple Markov Chain Simulations")
    plt.xlabel("Time Steps")
    plt.ylabel("States")
    plt.legend()
    plt.grid()
    plt.show()


def analyze_convergence(transition_matrix, initial_distribution, max_steps=1000):
    """Analyze the convergence to steady-state distribution over time."""
    current_distribution = initial_distribution.copy()
    distributions_over_time = [current_distribution]

    for _ in range(max_steps):
        current_distribution = current_distribution @ transition_matrix
        distributions_over_time.append(current_distribution)

        if np.allclose(current_distribution, distributions_over_time[-1]):
            break

    return distributions_over_time


def mean_first_passage_time(transition_matrix):
    """Compute mean first passage time using fundamental matrix."""
    I = np.eye(transition_matrix.shape[0])
    Q = transition_matrix.copy()

    N = np.linalg.inv(I - Q + np.eye(Q.shape[0]) * np.finfo(float).eps)

    return N


def state_visitation_frequency(transition_matrix, initial_state, num_steps):
    """Analyze state visitation frequencies over time."""
    counts = np.zeros(transition_matrix.shape[0])
    current_state = initial_state

    for _ in range(num_steps):
        counts[current_state] += 1
        current_state = np.random.choice(len(transition_matrix), p=transition_matrix[current_state])

    frequencies = counts / num_steps
    return frequencies


def simulate_and_visualize_chains(num_chains=5):
    """Simulate and visualize multiple chains with varying parameters."""
    for i in range(num_chains):
        random_transition_matrix = generate_random_stochastic_matrix(3)  # Randomize each time
        initial_states = [np.random.randint(3)]  # Random starting state from available states
        chains = simulate_multiple_chains(random_transition_matrix,
                                          initial_states * num_chains,
                                          num_steps=100)
        plot_multiple_chains(chains)


def menu():
    print("\nMarkov Chain Simulation Menu:")
    print("1. Simulate Markov Chain")
    print("2. Generate Random Stochastic Matrix")
    print("3. Plot Transition Matrix")
    print("4. Calculate Steady-State Distribution")
    print("5. Analyze Absorption Probabilities")
    print("6. Simulate Multiple Chains")
    print("7. Analyze Convergence to Steady-State")
    print("8. Compute Mean First Passage Times")
    print("9. Exit")


def main():
    while True:
        menu()

        try:
            choice = int(input("Select an option (1-9): "))

            if choice == 1:
                # Simulate Markov Chain
                initial_state = int(input("Enter initial state (0 or 1): "))
                num_steps = int(input("Enter number of steps: "))
                transition_matrix = np.array([[0.8, 0.2], [0.4, 0.6]])
                mc = MarkovChain(transition_matrix)
                states = mc.simulate(initial_state, num_steps)
                plot_markov_chain(states, mc.num_states)

            elif choice == 2:
                # Generate Random Stochastic Matrix
                size = int(input("Enter size of stochastic matrix: "))
                random_transition_matrix = generate_random_stochastic_matrix(size)
                print(f"Random Transition Matrix:\n{random_transition_matrix}")

            elif choice == 3:
                # Plot Transition Matrix
                transition_matrix = np.array([[0.8, 0.2], [0.4, 0.6]])
                plot_transition_matrix(transition_matrix)

            elif choice == 4:
                # Calculate Steady-State Distribution
                transition_matrix = np.array([[0.8, 0.2], [0.4, 0.6]])
                mc = MarkovChain(transition_matrix)
                steady_state_dist = mc.steady_state_distribution()
                print(f"Steady-state distribution: {steady_state_dist}")

            elif choice == 5:
                # Analyze Absorption Probabilities
                transition_matrix = np.array([[0.8, 0.2], [1.0, 0.0]])  # Example with absorbing state
                mc = MarkovChain(transition_matrix)
                absorption_probs = mc.absorption_probabilities()
                if absorption_probs is not None:
                    print(f"Absorption Probabilities:\n{absorption_probs}")

            elif choice == 6:
                # Simulate Multiple Chains
                num_steps = int(input("Enter number of steps: "))
                random_transition_matrix = generate_random_stochastic_matrix(3)  # 3x3 stochastic matrix
                initial_states = [0, 1]
                chains = simulate_multiple_chains(random_transition_matrix, initial_states, num_steps)
                plot_multiple_chains(chains)

            elif choice == 7:
                # Analyze Convergence to Steady-State
                initial_distribution = np.array([1.0, 0.0])  # Start in state 0
                transition_matrix = np.array([[0.8, 0.2], [0.4, 0.6]])
                distributions_over_time = analyze_convergence(transition_matrix, initial_distribution)

                plt.figure(figsize=(10, 6))
                for i in range(transition_matrix.shape[0]):
                    plt.plot([d[i] for d in distributions_over_time], label=f'State {i}')
                plt.title("Convergence to Steady-State Distribution")
                plt.xlabel("Iterations")
                plt.ylabel("Distribution Probability")
                plt.legend()
                plt.grid()
                plt.show()

            elif choice == 8:
                # Compute Mean First Passage Times
                transition_matrix = np.array([[0.8, 0.2], [1.0, 0.0]])  # Example with absorbing state
                mean_passage_times = mean_first_passage_time(transition_matrix)
                print(f"Mean First Passage Times:\n{mean_passage_times}")

            elif choice == 9:
                print("Exiting...")
                break

            else:
                print("Invalid option! Please select a valid option.")

        except ValueError as e:
            print(f"Input error: {e}. Please enter valid integers.")
        except Exception as e:
            print(f"An error occurred: {e}")
if __name__ == '__main__':
    main()