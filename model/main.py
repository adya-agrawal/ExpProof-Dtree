from model import load_credit_and_train, load_adult_and_train
from global_tree_surrogate import train_global_surrogate_tree, print_global_rules, compute_accuracies, export_tree_json, export_witness_json
from lime_shap import lime_fidelity, dtree_fidelity, shap_fidelity
import json

def run_pipeline(name, loader_fn):

    print("\n==============================")
    print(f" Running pipeline for {name}")
    print("==============================")

    # Load model + data
    model, X_train, y_train, X_test, y_test, input_dim, scaler, features = loader_fn()

    # Train global surrogate tree
    tree = train_global_surrogate_tree(model, X_train,
                                       max_depth=4,
                                       min_samples_leaf=200)

    print("Tree accuracies", compute_accuracies(model,tree, X_test, y_test))

    # -----------------------------------------------------
    # SAVE TREE JSON
    # -----------------------------------------------------
    tree_json = export_tree_json(tree, feature_names=features)

    with open(f"{name}_tree.json", "w") as f:
        json.dump(tree_json, f, indent=4)

    print(f"Saved {name}_tree.json ✓")

    # -----------------------------------------------------
    # SAVE WITNESS JSON (ExpProof-style)
    # -----------------------------------------------------
    x = X_test[0]              # pick first test point
    witness = export_witness_json(tree, x)

    with open(f"{name}_witness.json", "w") as f:
        json.dump(witness, f, indent=4)

    print(f"Saved {name}_witness.json ✓")

    print("\n=== ExpProof-Style Fidelity Evaluation ===")

    print("\n=== Fidelity (50 points) ===")

    # LIME Gaussian
    lg_mean, lg_std, lg_time = lime_fidelity(model, X_test, scaler, K=50, n_neigh=300, mode="gaussian")
    print(f"LIME Gaussian: {lg_mean:.4f} ± {lg_std:.4f} (time={lg_time:.2f}s)")

    # LIME Uniform
    lu_mean, lu_std, lu_time = lime_fidelity(model, X_test, scaler, K=50, n_neigh=300, mode="uniform")
    print(f"LIME Uniform:  {lu_mean:.4f} ± {lu_std:.4f} (time={lu_time:.2f}s)")

    # Decision Tree fidelity
    dt_mean, dt_std, dt_time = dtree_fidelity(model, tree, X_test, K=50)
    print(f"Decision Tree: {dt_mean:.4f} ± {dt_std:.4f} (time={dt_time:.4f}s)")

    # SHAP fidelity
    sh_mean, sh_std, sh_time = shap_fidelity(model, X_train, X_test, K=50, nsamples=300)
    print(f"SHAP:          {sh_mean:.4f} ± {sh_std:.4f} (time={sh_time:.2f}s)")

    print(f"Finished pipeline for {name} ✓\n")


# ---------------------------------------------------------
# Run both datasets
# ---------------------------------------------------------
run_pipeline("adult", load_adult_and_train)
run_pipeline("credit", load_credit_and_train)