# run_experiments.py
from model import load_credit_and_train
from global_tree_surrogate import (
    train_global_surrogate_tree,
    compute_accuracies,
    print_global_rules
)

print("\n=== Loading model and dataset ===")
model, X_train, y_train, X_test, y_test, input_dim = load_credit_and_train()

# ---------------------------------------------------------
# 1. Train global surrogate decision tree
# ---------------------------------------------------------
tree = train_global_surrogate_tree(
    model,
    X_train,                # you can also use X_train or whole X
    max_depth=2,
    min_samples_leaf=500,
    criterion="gini"
)

# ---------------------------------------------------------
# 2. Compute accuracies
# ---------------------------------------------------------
print("\n=== Accuracy Comparison ===")

nn_acc, tree_acc, fidelity = compute_accuracies(model, tree, X_test, y_test)

print(f"Neural Network accuracy:     {nn_acc:.4f}")
print(f"Global Decision Tree accuracy {tree_acc:.4f}")
print("NN-tree agreement:", fidelity)

# ---------------------------------------------------------
# 3. Print global explanation
# ---------------------------------------------------------
print_global_rules(tree, input_dim)


