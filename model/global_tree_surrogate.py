# global_tree_surrogate.py

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text


# ---------------------------------------------------------
# 1. Helper: predict NN outputs on numpy arrays
# ---------------------------------------------------------
def nn_predict(model, X_np):
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32)
        logits = model(X_t)
        return logits.argmax(dim=1).cpu().numpy()



# ---------------------------------------------------------
# 2. Train global surrogate decision tree
#    - Train on the entire training or test set
#    - Uses NN predictions as labels
# ---------------------------------------------------------
def train_global_surrogate_tree(model,
                                X,
                                max_depth=6,
                                min_samples_leaf=20,
                                criterion="gini"):
    """
    Train a global decision tree that mimics the neural network.

    Inputs:
        model: PyTorch NN
        X: numpy array of shape (N, d) - dataset features
        max_depth: tree depth
        min_samples_leaf: smoothing
        criterion: "gini" or "entropy"

    Returns:
        trained sklearn DecisionTreeClassifier
    """
    print("Generating NN pseudo-labels...")
    y_nn = nn_predict(model, X)

    print("Training global surrogate decision tree...")
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=0
    )
    tree.fit(X, y_nn)

    return tree



# ---------------------------------------------------------
# 3. Compute validation accuracy of NN and tree
# ---------------------------------------------------------
def compute_accuracies(model, tree, X_test, y_test):
    """
    Returns:
        nn_val_acc, tree_val_acc
    """
    y_nn_pred = nn_predict(model, X_test)
    y_tree_pred = tree.predict(X_test)

    nn_val_acc = (y_nn_pred == y_test).mean()
    tree_val_acc = (y_tree_pred == y_test).mean()
    fidelity = (y_nn_pred == y_tree_pred).mean()

    return nn_val_acc, tree_val_acc, fidelity



# ---------------------------------------------------------
# 4. Human-readable global explanation (rules)
# ---------------------------------------------------------
def print_global_rules(tree, input_dim):
    feature_names = [f"x{i}" for i in range(input_dim)]
    rules = export_text(tree, feature_names=feature_names)
    print("\n=== Global Decision Tree Rules ===")
    print(rules)
    return rules
