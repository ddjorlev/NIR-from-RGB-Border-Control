# RGB2NIR Model Evaluation - Face Recognition Verification

This repository provides the evaluation framework for downstream face recognition (verification) tasks using your RGB2NIR generated images. 

## Overview
The goal is to evaluate the performance of your RGB2NIR models using a custom evaluation protocol on the DLORD dataset - Protocol A (Face Verification - determining if two videos show the same person or not). You will take an existing dataset of RGB videos, convert them to synthetic NIR using your models, extract face embeddings using the provided scripts, and evaluate the verification performance. 

The core hypothesis is straightforward: **if your RGB2NIR translation model preserves identity features well, verifying synthetic NIR against original NIR should yield better performance than a baseline comparison of raw RGB to original NIR.**

### Protocol A: Video Verification
The provided scripts handle the entire underlying evaluation pipeline. For your understanding, Protocol A evaluates the following:
- **Media**: A single video identified by a Media ID (`mid`).
- **Frames**: Each media may have multiple frames, identified by Frame Indices (`fid`).
- **Templates**: Built by aggregating all frame embeddings across all media corresponding to a specific template ID into a single, compact biometric template.
- **Evaluation**: Pairs of templates are compared (same person vs. different person). The comparison is **always cross-spectral**. For your evaluation, this means you will be comparing your **Synthetic NIR** templates against **Original NIR** templates, which will hopefully yield a stronger verification result than the raw RGB baseline.
- **Metrics**: TPR@FPR (True Positive Rate at a given False Positive Rate ranging from 1e-1 down to 1e-5), computed using stratified bootstrapping for exact uncertainty quantification (95% confidence intervals).

> [!CRITICAL]
> - **Image Dimensions**: Your generated outputs must be exactly **112x112** NIR images.
> - **Directory Structure**: You must perfectly preserve the folder structure of the original dataset (see the provided `DLORD_synthetic_nir` example).
> - **Feature Extractor Consistency**: Always extract your embeddings using the same underlying model backbone across the entire test matrix (e.g., use AdaFace across the board, or ArcFace across the board). **Comparing cross-model embeddings (such as AdaFace vs. ArcFace) will completely destroy your performance.**

---

## Directory Structure & Project Layout

1. **`DLORD_rgb2nir/`**: Contains the raw input data (original dataset) for your model.
   - Folders correspond to unique subject identities.
   - Inside each identity folder, subfolders correspond to distinct video clips. The vast majority of the time there are three RGB video folders (the ones you need to convert to NIR) starting with `rgbn` and one original NIR video folder (this is the ground truth from the dataset) starting with `irn`. All videos of the same identity were taken in different, non-overlapping scenes.
   
2. **`DLORD_synthetic_nir/`** (Your Task!):
   - You need to run your trained RGB2NIR model (e.g., Pix2Pix GAN) on the images inside `DLORD_rgb2nir/`.
   - Save the generated synthetic NIR images to a separate folder (like I did with this one).
   - **Crucial:** You must perfectly replicate the exact file names and directory hierarchy from `DLORD_rgb2nir/`. 
   - *Example note: An example is already provided where a Pix2Pix GAN was trained from scratch on the Tufts Faces RGB-NIR dataset (also provided) and all rgb videos from DLORD_rgb2nir were converted to nir and saved here with the same directory structure.*

3. **`dlord_embeddings/`**:
   - This folder stores the extracted `.pt` embedding files for the original RGB, original NIR, and your custom synthetic NIR images. 
   - The extraction pipeline utilizes caching; embeddings for the original reference NIR videos only need to be computed once. When you generate a new folder of synthetic NIR images, only your model's outputs will be processed.

4. **Models & Scripts**:
   - `converted_adaface_ir101_ms1mv3.pt` & `converted_arcface_r100_ms1mv3.pt`: Pretrained face recognition models used to extract biometric embeddings.
   - `01_extract_embeddings.sh`: Script to extract face embeddings. It uses caching and will only recompute what is necessary. When you add a new folder of synthetic NIR images, just add a line to this script pointing to it and run it to cache the new embeddings.
   - `02_eval_protocol_A.sh`: Script to evaluate Protocol A using the extracted embeddings. It takes in two embedding targets (your synthetic NIR embeddings vs. the original reference NIR embeddings).
5. **tufts_faces_rgb_nir**:
   - This is a serparate dataset that we managed to obtain which has aligned RGB and NIR images (mostly, not exactly). You can use this for finetuning or training from scratch your RGB2NIR model. The images in `DLORD_synthetic_nir` were produced by a pix2pix gan that was trained from scratch only on this dataset.


## Python env setup

`requirements.txt` is provided, but in general you need: torch, torchvision, pandas, tqdm


## Step-by-Step Guide

### Step 1: Generate Synthetic NIR
Convert all the original RGB images in `DLORD_rgb2nir/` to synthetic NIR images using your trained generator. 
Save the outputs to your designated folder, ensuring you maintain the exact directory structure of the inputs. I have included an example of what this should look like in `DLORD_synthetic_nir/`.

### Step 2: Extract Embeddings
Once you have generated all synthetic NIR images, extract the biometric embeddings for them.
The provided script uses pre-trained face recognition backbones (AdaFace and ArcFace) and saves the outputs to `dlord_embeddings/`.

Run the extraction script:
```bash
./01_extract_embeddings.sh

```

### Step 3: Evaluate Protocol A

After the embeddings are extracted and saved, you can run the evaluation. The evaluation script compares the cross-spectral pairs of templates (Synthetic NIR vs. Original NIR) and outputs the verification metrics. Evaluation is also done on the GPU for a major speedup (on an RTX3090 a single evaluation takes ~1min)

Run the evaluation script:

```bash
/02_eval_protocol_A.sh > example_results.txt

```


The script will compute TPR@FPR using optimized, hardware-flexible bootstrapping (leveraging your GPU if available, or falling back safely to multi-threaded CPU sorting if not) and output the final verification performance and 95% confidence intervals.

I have included the results of the evaluation with the original rgb2nir and example DLORD_synthetic_nir images (see example_results.txt). As you can see the synthetic_nir2nir results are worse than rgb2nir. Hopefully you have better models.

*Note: You do not need to (and should not) modify any underlying protocol `.csv` metadata files.*

