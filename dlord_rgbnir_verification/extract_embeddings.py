from fr_model import get_adaface_ir101_model, get_arcface_ir101_model
import pandas as pd
from pathlib import Path
import torch
import torchvision
import torchvision.transforms.v2.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import argparse


class DLORDDataset(Dataset):
    def __init__(self, df, root_dir, transform):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["filename"]
        img = torchvision.io.decode_image(img_path)
        img = TF.to_dtype(img, torch.float, scale=True)
        return {"img": img, "mid": row["mid"], "fid": row["fid"]}


def main():
    parser = argparse.ArgumentParser(
        description="Extract face embeddings from a specific image directory."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the directory containing the images.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where the output .pt file should be saved.",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["real_rgb", "real_nir", "synthetic_nir"],
        help="Semantic label for the dataset. Handles internal CSV mapping automatically.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["adaface", "arcface"],
        help="Select the feature extraction model.",
    )

    # --- The Required Tag Argument ---
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Custom identifier for this run (e.g., v1, baseline, pix2pix_epoch50).",
    )
    # ---------------------------------

    parser.add_argument(
        "--mids_csv",
        type=str,
        default="dlord_flat_mids_fids.csv",
        help="Path to the mapping CSV.",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()

    fr_model, inp_transform, model_tag = None, None, None
    if args.model == "adaface":
        fr_model, inp_transform, model_tag = get_adaface_ir101_model()
    elif args.model == "arcface":
        fr_model, inp_transform, model_tag = get_arcface_ir101_model()
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    inp_transform = inp_transform.cuda()
    fr_model = fr_model.cuda().eval()

    out_dir = Path(args.out_dir)
    root_path = Path(args.root_dir)

    # Injecting the user tag into the final filename
    out_filename = f"{args.dataset_type}_{model_tag}_{args.tag}_mid_fid2emb.pt"
    out_path = out_dir / out_filename

    print("=" * 50)
    print(f"RUNNING EXTRACTION: {out_filename}")
    print(f"MODEL: {args.model.upper()} | TAG: {args.tag}")
    print("=" * 50)

    if out_path.exists():
        print(f"\n[CACHE HIT] Skipping. File already exists: {out_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_type in ["real_rgb", "synthetic_nir"]:
        internal_camera_filter = "rgbn_08"
    else:
        internal_camera_filter = "irn_09"

    if not os.path.exists(args.mids_csv):
        print(f"Error: Could not find CSV file at {args.mids_csv}")
        return

    df = pd.read_csv(args.mids_csv)

    sub_df = df[df["camera"] == internal_camera_filter].reset_index(drop=True)

    exists_mask = sub_df["filename"].apply(lambda p: (root_path / p).exists())
    valid_df = sub_df[exists_mask].reset_index(drop=True)

    print(f"Found {len(valid_df)} valid images to process in {root_path.name}")
    if len(valid_df) == 0:
        print("No valid images found. Exiting.")
        return

    dataset = DLORDDataset(valid_df, root_path, inp_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    mid_fid2emb = {}

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Extracting"):
            imgs = batch["img"].cuda()
            imgs = inp_transform(imgs)
            mids = batch["mid"]
            fids = batch["fid"]

            out = fr_model(imgs)
            embs = out.embedding.cpu()
            norms = out.norm.cpu()

            embs = embs * norms.unsqueeze(1)

            for i in range(len(mids)):
                key = (mids[i], fids[i].item())
                mid_fid2emb[key] = embs[i].numpy()

    torch.save(mid_fid2emb, out_path)
    print(f"\nSuccessfully saved features to {out_path}")


if __name__ == "__main__":
    main()
