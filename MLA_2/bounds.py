"""
import random
import matplotlib.pyplot as plt
import numpy as np

results = [sum([random.randint(0, 1) for _ in range(20)]) for _ in range(1000000)]  # Run experiment
results.sort()  # Sort the results in ascending order

probs = []

for num in range(10, 21):
    prob = len([x for x in results if x >= num]) / 1000000
    probs.append(prob)

E = np.mean(results)    
Var = np.var(results)
print(E)
print(Var)


plt.figure(figsize=(10, 6))

plt.plot(range(10, 21), probs, marker='o', label="Actual Probability")
plt.plot(range(10, 21), [E/ x if E / x <= 1 else 1 for x in range(10, 21)], marker='o', label="Markov's Bound")
plt.plot(range(10, 21), [Var / (x * x) for x in range(10, 21)], marker='o', label="Chebyshev's Bound")
plt.xlabel("Number of Successes (k)")
plt.ylabel("Probability")
plt.title("Probability of Having k or More Successes")
plt.legend()
plt.grid(True)

plt.show()
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# Number of repetitions and experiments
num_repetitions = 1000000
num_experiments = 20

# List to store the empirical frequencies
empirical_frequencies = []

# Range of alpha values
alpha_values = [0.5 + 0.05 * i for i in range(10)]

for alpha in alpha_values:
    count = 0  # Counter for successful experiments

    for _ in range(num_repetitions):
        # Simulate the experiment by drawing 20 Bernoulli random variables
        outcomes = [random.random() <= 0.5 for _ in range(num_experiments)]
        
        # Check if the fraction of successes is greater than or equal to alpha
        if sum(outcomes) / num_experiments >= alpha:
            count += 1

    empirical_frequency = count / num_repetitions
    empirical_frequencies.append(empirical_frequency)

# Calculate Markov's, Chebyshev's, and Hoeffding's bounds
markov_bounds = [0.5 / alpha for alpha in alpha_values]
chebyshev_bounds = [0.25 / (alpha * alpha) for alpha in alpha_values]
hoeffding_bounds = [min(2 * np.exp(-2 * num_experiments * (alpha - 0.5)**2),1) for alpha in alpha_values]

# Plot the results
plt.figure(figsize=(10, 6))

plt.plot(alpha_values, empirical_frequencies, marker='o', label="Empirical Frequency")
plt.plot(alpha_values, markov_bounds, marker='o', label="Markov's Bound")
plt.plot(alpha_values, chebyshev_bounds, marker='o', label="Chebyshev's Bound")
plt.plot(alpha_values, hoeffding_bounds, marker='o', label="Hoeffding's Bound")

plt.xlabel("Alpha (α)")
plt.ylabel("Probability")
plt.title("Comparison of Empirical Frequencies and Bounds")
plt.legend()
plt.grid(True)

plt.show()

print ("prob alpha = 1",empirical_frequencies[9], "alpha = 0.95", empirical_frequencies[8])


# List to store the empirical frequencies
empirical_frequencies = []

# Range of alpha values
alpha_values = [0.1 + 0.05 * i for i in range(18)]

for alpha in alpha_values:
    count = 0  # Counter for successful experiments

    for _ in range(num_repetitions):
        # Simulate the experiment by drawing 20 Bernoulli random variables with bias 0.1
        outcomes = [random.random() <= 0.1 for _ in range(num_experiments)]
        
        # Check if the fraction of successes is greater than or equal to alpha
        if sum(outcomes) / num_experiments >= alpha:
            count += 1

    empirical_frequency = count / num_repetitions
    empirical_frequencies.append(empirical_frequency)

print (empirical_frequencies)

# Calculate Markov's, Chebyshev's, and Hoeffding's bounds
markov_bounds = [min(0.1 / alpha, 1) for alpha in alpha_values]
chebyshev_bounds = [min(0.09 / (alpha * alpha), 1) for alpha in alpha_values]
hoeffding_bounds = [min(2 * np.exp(-2 * num_experiments * (alpha - 0.1)**2), 1) for alpha in alpha_values]

# Plot the results
plt.figure(figsize=(10, 6))

plt.plot(alpha_values, empirical_frequencies, marker='o', label="Empirical Frequency")
plt.plot(alpha_values, markov_bounds, marker='o', label="Markov's Bound")
plt.plot(alpha_values, chebyshev_bounds, marker='o', label="Chebyshev's Bound")
plt.plot(alpha_values, hoeffding_bounds, marker='o', label="Hoeffding's Bound")

plt.xlabel("Alpha (α)")
plt.ylabel("Probability")
plt.title("Comparison of Empirical Frequencies and Bounds (Bias = 0.1)")
plt.legend()
plt.grid(True)

plt.show()