import scanpy as sc
import pandas as pd
import time
import sys
import os
import glob
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pyscDblFinder.scDblFinder import compute_doublet_score

def eval_mode(adata, mode_name, clusters_col, ds_name, use_gpu=False):
    st = time.time()
    try:
        adata_res = compute_doublet_score(adata.copy(), n_iters=3, random_state=42,
                                          clusters_col=clusters_col, use_gpu=use_gpu)
    except Exception as e:
        print(f"    Failed on {mode_name} for {ds_name}: {e}")
        return None

    elapsed = time.time() - st

    truth_labels = (adata_res.obs['truth'] == 'doublet').astype(int)
    scores = adata_res.obs['scDblFinder_score']

    auroc = roc_auc_score(truth_labels, scores)
    precision, recall, _ = precision_recall_curve(truth_labels, scores)
    auprc = auc(recall, precision)

    return {
        "dataset": ds_name,
        "method": mode_name,
        "AUPRC": auprc,
        "AUROC": auroc,
        "elapsed": elapsed
    }

def main():
    use_gpu = '--gpu' in sys.argv
    datasets = glob.glob(os.path.join(os.path.dirname(__file__), "datasets", "*.h5ad"))
    all_results = []

    print(f"Running benchmark (use_gpu={use_gpu})...")
    for ds in datasets:
        ds_name = os.path.basename(ds).replace(".h5ad", "")
        print(f"Evaluating {ds_name}...")

        adata = sc.read_h5ad(ds)
        # some datasets string parsing
        if 'truth' in adata.obs:
            adata.obs['truth'] = adata.obs['truth'].str.lower()

        print(f"  -> Clustered Mode")
        res_clust = eval_mode(adata, "scDblFinder.Py.clusters", "clusters", ds_name,
                             use_gpu=use_gpu)
        if res_clust:
            all_results.append(res_clust)

        print(f"  -> Random Mode")
        res_rand = eval_mode(adata, "scDblFinder.Py.random", None, ds_name,
                             use_gpu=use_gpu)
        if res_rand:
            all_results.append(res_rand)
            
    df = pd.DataFrame(all_results)
    out_path = os.path.join(os.path.dirname(__file__), "python_benchmark_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved python benchmark scores across all datasets to {out_path}")

if __name__ == "__main__":
    main()
