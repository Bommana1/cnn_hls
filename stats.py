import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Define the directory where NIST results are stored
nist_results_dir = "path_to_nist_results"  # Change this to your actual path

# Regular expression to extract test names and p-values
p_value_pattern = re.compile(r"(\w+)\s+P-value\s+=\s+([\d.]+)")

# Store results
test_results = {}

# Iterate over NIST output files
for filename in os.listdir(nist_results_dir):
    if filename.endswith(".txt"):  # Ensure it's a result file
        with open(os.path.join(nist_results_dir, filename), "r") as file:
            for line in file:
                match = p_value_pattern.search(line)
                if match:
                    test_name, p_value = match.groups()
                    p_value = float(p_value)

                    if test_name not in test_results:
                        test_results[test_name] = []
                    test_results[test_name].append(p_value)

# Summarizing Pass/Fail
alpha = 0.01  # Typical threshold for statistical significance
summary = {test: sum(1 for p in p_values if p > alpha) / len(p_values)
           for test, p_values in test_results.items()}

# Convert summary to lists for plotting
tests = list(summary.keys())
pass_rates = list(summary.values())

# Plot results
plt.figure(figsize=(12, 6))
plt.barh(tests, pass_rates, color="skyblue")
plt.xlabel("Pass Rate (%)")
plt.ylabel("NIST Test")
plt.title("NIST Randomness Test Summary")
plt.xlim(0, 1)
plt.grid(axis="x", linestyle="--", alpha=0.6)

# Show percentages on bars
for index, value in enumerate(pass_rates):
    plt.text(value, index, f"{value:.2%}", va='center', fontsize=10)

plt.show()
