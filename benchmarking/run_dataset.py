import sys
import os
import scanpy as sc
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pyscDblFinder.scDblFinder import compute_doublet_score

def eval_mode(adata, mode_name, clusters_col, ds_name, n_repeats=1, use_gpu=False):
    aurocs = []
    auprcs = []
    elapsed_times = []

    for i in range(n_repeats):
        st = time.time()
        try:
            # Use a fixed seed and fixed number of iterations to match the
            # full-benchmark behaviour and ensure reproducible, comparable
            # results across runs.
            adata_res = compute_doublet_score(
                adata.copy(),
                random_state=42,
                n_iters=3,
                clusters_col=clusters_col,
                n_features=1000,
                use_gpu=use_gpu,
            )
        except NotImplementedError as e:
            print(f"    Failed on {mode_name} for {ds_name} (repeat {i}): {e}")
            continue

        elapsed = time.time() - st
        truth_labels = (adata_res.obs['truth'] == 'doublet').astype(int)
        scores = adata_res.obs['scDblFinder_score']
        
        auroc = roc_auc_score(truth_labels, scores)
        precision, recall, _ = precision_recall_curve(truth_labels, scores)
        auprc = auc(recall, precision)
        
        aurocs.append(auroc)
        auprcs.append(auprc)
        elapsed_times.append(elapsed)
        
    if not aurocs:
        return None
        
    return {
        "dataset": ds_name,
        "method": mode_name,
        "AUPRC": sum(auprcs) / len(auprcs),
        "AUROC": sum(aurocs) / len(aurocs),
        "elapsed": sum(elapsed_times) / len(elapsed_times)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_dataset.py <dataset_name> [n_repeats] [--gpu]")
        sys.exit(1)

    ds_name = sys.argv[1]
    remaining = sys.argv[2:]
    use_gpu = '--gpu' in remaining
    remaining = [a for a in remaining if a != '--gpu']
    n_repeats = int(remaining[0]) if remaining else 1
    n_features = 1000
    ds_path = f"datasets/{ds_name}.h5ad"
    if not os.path.exists(ds_path):
        print(f"Dataset {ds_path} not found.")
        sys.exit(1)

    print(f"Evaluating {ds_name} (use_gpu={use_gpu})...")
    adata = sc.read_h5ad(ds_path)
    if 'truth' in adata.obs:
        adata.obs['truth'] = adata.obs['truth'].str.lower()

    all_results = []
    print(f"  -> Clustered Mode")
    res_clust = eval_mode(adata, "scDblFinder.Py.clusters", "clusters", ds_name,
                          n_repeats=n_repeats, use_gpu=use_gpu)
    if res_clust:
        all_results.append(res_clust)

    print(f"  -> Random Mode")

    aurocs = []
    auprcs = []
    elapsed_times = []

    for i in range(n_repeats):
        st = time.time()
        try:
            # Match the full-benchmark: fixed seed and fixed iteration count.
            adata_res = compute_doublet_score(
                adata.copy(),
                random_state=42,
                n_iters=3,
                clusters_col=None,
                n_features=n_features,
                use_gpu=use_gpu,
            )
        except NotImplementedError as e:
            print(f"Failed on random mode for {ds_name}: {e}")
            continue

        elapsed = time.time() - st
        truth_labels = (adata_res.obs['truth'] == 'doublet').astype(int)
        scores = adata_res.obs['scDblFinder_score']
        
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        aurocs.append(roc_auc_score(truth_labels, scores))
        precision, recall, _ = precision_recall_curve(truth_labels, scores)
        auprcs.append(auc(recall, precision))
        elapsed_times.append(elapsed)

    if aurocs:
        res_rand = {
            "dataset": ds_name,
            "method": "scDblFinder.Py.random",
            "AUPRC": sum(auprcs) / len(auprcs),
            "AUROC": sum(aurocs) / len(aurocs),
            "elapsed": sum(elapsed_times) / len(elapsed_times)
        }
        all_results.append(res_rand)
        
    df = pd.DataFrame(all_results)
    out_path = f"python_benchmark_{ds_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved python benchmark scores for {ds_name} to {out_path}")

if __name__ == "__main__":
    main()