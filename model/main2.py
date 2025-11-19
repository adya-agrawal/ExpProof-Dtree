# exp_layers.py
import torch
import torch.optim as optim
from dynamic_net import DynamicNet
from model import load_credit_and_train
from global_tree_surrogate import train_global_surrogate_tree, compute_accuracies
from lime_shap import lime_fidelity, dtree_fidelity, shap_fidelity

# Networks to test: total layers
L_LIST = [2, 4, 8, 16]

results = {}

# Load dataset once
_, X_train, y_train, X_test, y_test, input_dim, scaler = load_credit_and_train(epochs=0)

for L in L_LIST:

    print(f"\n==============================")
    print(f" Training Neural Network with {L} layers")
    print(f"==============================")

    # Build dynamic network
    model = DynamicNet(
        input_dim=input_dim,
        hidden_size=16,
        output_dim=2,
        total_layers=L
    )

    # Train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(200):  # fewer epochs -> still clean for comparison
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    # Train surrogate decision tree
    tree = train_global_surrogate_tree(model, X_train,
                                       max_depth=4,
                                       min_samples_leaf=200)

    # Fidelity scores
    lime_g, _, _ = lime_fidelity(model, X_test, scaler, K=50, n_neigh=300, mode="gaussian")
    lime_u, _, _ = lime_fidelity(model, X_test, scaler, K=50, n_neigh=300, mode="uniform")
    dt_fid, _, _ = dtree_fidelity(model, tree, X_test, K=50)
    sh_fid, _, _ = shap_fidelity(model, X_train, X_test, K=50, nsamples=300)

    # Accuracies
    accuracies = compute_accuracies(model, tree, X_test, y_test)

    results[L] = {
        "NN accuracy": accuracies[0],
        "Tree accuracy": accuracies[1],
        "LIME Gaussian": lime_g,
        "LIME Uniform":  lime_u,
        "Tree fidelity": dt_fid,
        "SHAP fidelity": sh_fid
    }

print("\n\n=========== FINAL TABLE ===========")
for L in L_LIST:
    print(f"\nLayers = {L}")
    for k, v in results[L].items():
        print(f"{k}: {v:.4f}")
