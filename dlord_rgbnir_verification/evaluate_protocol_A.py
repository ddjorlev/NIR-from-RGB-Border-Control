import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
BOOTSTRAP_ITERATIONS = 10000
FPR_TARGETS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def set_seed(seed):
    """Locks all random number generators for exact reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_templates(merged_df, embedding_dict):
    """Aggregates raw embeddings and L2 normalizes them per template_id."""
    bags = defaultdict(list)

    # Fast iteration using itertuples
    for row in tqdm(
        merged_df[["template_id", "mid", "fid"]].itertuples(index=False),
        total=len(merged_df),
        desc="Mapping",
        leave=False,
    ):
        key = (row.mid, int(row.fid))
        if key in embedding_dict:
            bags[row.template_id].append(embedding_dict[key])

    # Fast tensor aggregation
    template_signatures = {}
    for tid, emb_list in tqdm(bags.items(), desc="Normalizing", leave=False):
        bag_tensor = torch.from_numpy(np.stack(emb_list))
        template_raw = torch.sum(bag_tensor, dim=0)
        template_signatures[tid] = torch.nn.functional.normalize(
            template_raw, p=2, dim=0
        )

    return template_signatures


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cross-Modal Verification")
    parser.add_argument(
        "--source_embs",
        required=True,
        help="Path to the source (RGB or Synthetic NIR) .pt file",
    )
    parser.add_argument(
        "--target_embs",
        required=True,
        help="Path to the target (Original NIR) .pt file",
    )
    parser.add_argument("--templates_csv", default="protocol_A_agnostic_templates.csv")
    parser.add_argument("--pairs_csv", default="protocol_A_agnostic_pairs.csv")
    parser.add_argument("--mids_csv", default="dlord_flat_mids_fids.csv")
    parser.add_argument("--experiment_name", default="Evaluation Run")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible bootstrapping",
    )
    parser.add_argument(
        "--save_npz",
        default=None,
        help="Optional path to save raw genuine/impostor scores + bootstrap "
             "results as .npz, for building ROC curves and comparison plots "
             "across experiments/models later. Does not change stdout output.",
    )
    args = parser.parse_args()

    # Apply reproducibility seed
    set_seed(args.seed)

    # Hardware fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print(f"RUNNING: {args.experiment_name}")
    print(f"HARDWARE: {str(device).upper()} | SEED: {args.seed}")
    print("=" * 50)

    print("Loading datasets...")
    source_dict = torch.load(args.source_embs, weights_only=False)
    target_dict = torch.load(args.target_embs, weights_only=False)

    mids_df = pd.read_csv(args.mids_csv)
    templates_df = pd.read_csv(args.templates_csv)
    pairs_df = pd.read_csv(args.pairs_csv)

    merged_df = templates_df.merge(
        mids_df[["filename", "mid", "fid"]], on="filename", how="left"
    )

    print("Processing source templates...")
    source_templates = build_templates(merged_df, source_dict)

    print("Processing target templates...")
    target_templates = build_templates(merged_df, target_dict)

    print("Computing cosine similarities (Vectorized)...")

    valid_sources = set(source_templates.keys())
    valid_targets = set(target_templates.keys())

    valid_pairs = pairs_df[
        pairs_df["template_id_1"].isin(valid_sources)
        & pairs_df["template_id_2"].isin(valid_targets)
    ]

    missing_pairs = len(pairs_df) - len(valid_pairs)
    if missing_pairs > 0:
        print(f"Warning: Dropped {missing_pairs} pairs due to missing embeddings.")

    unique_t1 = list(valid_sources)
    t1_to_idx = {t: i for i, t in enumerate(unique_t1)}
    source_mat = torch.stack([source_templates[t] for t in unique_t1])

    unique_t2 = list(valid_targets)
    t2_to_idx = {t: i for i, t in enumerate(unique_t2)}
    target_mat = torch.stack([target_templates[t] for t in unique_t2])

    idx1 = torch.tensor(valid_pairs["template_id_1"].map(t1_to_idx).values)
    idx2 = torch.tensor(valid_pairs["template_id_2"].map(t2_to_idx).values)
    labels = valid_pairs["label"].values

    CHUNK_SIZE = 1000000
    scores = torch.zeros(len(valid_pairs), dtype=torch.float32)

    for i in tqdm(range(0, len(valid_pairs), CHUNK_SIZE), desc="Scoring Batches"):
        batch_idx1 = idx1[i : i + CHUNK_SIZE]
        batch_idx2 = idx2[i : i + CHUNK_SIZE]

        batch_scores = torch.sum(source_mat[batch_idx1] * target_mat[batch_idx2], dim=1)
        scores[i : i + CHUNK_SIZE] = batch_scores

    scores_np = scores.numpy()
    genuine_scores = scores_np[labels == 1]
    impostor_scores = scores_np[labels == 0]

    print(
        f"\nScore Arrays Built: {len(genuine_scores)} Genuine | {len(impostor_scores)} Impostor"
    )

    # --- Stratified Bootstrapping (Hardware Agnostic) ---
    print(
        f"\nRunning Stratified Bootstrap ({BOOTSTRAP_ITERATIONS} iterations) on {str(device).upper()}..."
    )

    bootstrap_results = {fpr: [] for fpr in FPR_TARGETS}

    # Move to the selected device
    gen_tensor = torch.tensor(genuine_scores, dtype=torch.float32, device=device)
    imp_tensor = torch.tensor(impostor_scores, dtype=torch.float32, device=device)
    q_targets = torch.tensor(
        [1.0 - fpr for fpr in FPR_TARGETS], dtype=torch.float32, device=device
    )

    for _ in tqdm(range(BOOTSTRAP_ITERATIONS), desc="Bootstrapping", leave=False):
        # Fast device-native sampling
        gen_idx = torch.randint(0, len(gen_tensor), (len(gen_tensor),), device=device)
        imp_idx = torch.randint(0, len(imp_tensor), (len(imp_tensor),), device=device)

        gen_sample = gen_tensor[gen_idx]
        imp_sample = imp_tensor[imp_idx]

        # Compute all quantiles at once
        thresholds = torch.quantile(imp_sample, q_targets)

        for i, fpr in enumerate(FPR_TARGETS):
            tpr = (gen_sample >= thresholds[i]).float().mean().item()
            bootstrap_results[fpr].append(tpr)

    # --- Final Report ---
    print("\n" + "=" * 50)
    print("FINAL RESULTS (95% Confidence Intervals)")
    print("=" * 50)

    for fpr in FPR_TARGETS:
        tpr_array = np.array(bootstrap_results[fpr]) * 100
        mean_tpr = np.mean(tpr_array)
        lower_bound = np.percentile(tpr_array, 2.5)
        upper_bound = np.percentile(tpr_array, 97.5)
        fpr_str = f"1e-{int(abs(np.log10(fpr)))}"
        print(
            f"TPR @ FPR {fpr_str}: \t {mean_tpr:.2f}%  [{lower_bound:.2f}% - {upper_bound:.2f}%]"
        )

    print("=" * 50)

    if args.save_npz:
        summary_fprs = np.array(FPR_TARGETS)
        summary_mean_tpr = np.array([
            np.mean(np.array(bootstrap_results[fpr]) * 100) for fpr in FPR_TARGETS
        ])
        summary_lower = np.array([
            np.percentile(np.array(bootstrap_results[fpr]) * 100, 2.5) for fpr in FPR_TARGETS
        ])
        summary_upper = np.array([
            np.percentile(np.array(bootstrap_results[fpr]) * 100, 97.5) for fpr in FPR_TARGETS
        ])
        np.savez(
            args.save_npz,
            experiment_name=args.experiment_name,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            fpr_targets=summary_fprs,
            mean_tpr=summary_mean_tpr,
            ci_lower=summary_lower,
            ci_upper=summary_upper,
        )
        print(f"Saved raw scores + summary to {args.save_npz}")


if __name__ == "__main__":
    main()
