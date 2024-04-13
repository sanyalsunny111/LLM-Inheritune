
import json
from collections import defaultdict

# Load the JSON file
with open("/out/results/ollama3b/results_mmlu.json", "r") as file:
    data = json.load(file)

# Extract the results
results = data['results']

# Compute the average accuracy for all tasks
all_accuracies = [task_data['acc'] for task_data in results.values()]
average_acc = sum(all_accuracies) / len(all_accuracies)
total_tasks = len(all_accuracies)
print(f"Average accuracy for all tasks: {average_acc:.4f}")
print(f"Number of all tasks: {total_tasks:.4f}")

# Compute the average accuracy norm for all tasks
all_accuracies_norm = [task_data['acc_norm'] for task_data in results.values()]
average_accnorm = sum(all_accuracies_norm) / len(all_accuracies_norm)
print(f"Average accuracy norm for all tasks: {average_accnorm:.4f}")
