# import json
# from collections import defaultdict
#
# # Load the JSON file
# with open("/scratch/07946/ss95332/out/results_mmlu.json", "r") as file:
#     data = json.load(file)
#
# # Extract the results
# results = data['results']
#
# # Compute the average accuracy for all tasks
# all_accuracies = [task_data['acc'] for task_data in results.values()]
# average_acc = sum(all_accuracies) / len(all_accuracies)
# print(f"Average accuracy for all tasks: {average_acc:.4f}")
#
# # Categorize tasks and calculate average accuracy for each category
# category_accuracies = defaultdict(list)
#
# for task, task_data in results.items():
#     category = task.split('-')[0]
#     category_accuracies[category].append(task_data['acc'])
#
# # Calculate and display average accuracy for each category
# for category, accs in category_accuracies.items():
#     avg_acc = sum(accs) / len(accs)
#     print(f"Average accuracy for category '{category}': {avg_acc:.4f}")

import json
from collections import defaultdict

# Load the JSON file
with open("/scratch/07946/ss95332/out/results/ollama3b/results_mmlu_191k.json", "r") as file:
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