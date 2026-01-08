import sys
import os
# WORKAROUND: Add this line to address the OpenMP runtime error.
# This is a common issue on Windows when using libraries like NumPy and PyTorch together.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import tifffile
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from torch_fidelity import calculate_metrics

def main(args):
    """
    Main function to run the evaluation.
    This script compares a folder of generated images against a folder of ground truth images.
    """
    # --- 1. Setup and Initialization ---
    print("--- Initializing Evaluation Script ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize the LPIPS model once. It will be used for per-image comparison.
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    except Exception as e:
        print(f"Error initializing LPIPS model: {e}")
        sys.exit(1)

    # --- 2. Prepare File Lists ---
    output_dir = Path(args.output_dir)
    gt_dir = Path(args.gt_dir)

    # Find all generated images (.png) and ground truth images (.tif/.tiff)
    # The lists are sorted to ensure that files with the same name are compared.
    output_files = sorted([f for f in output_dir.iterdir() if f.suffix.lower() == '.png'])
    gt_files = sorted([f for f in gt_dir.iterdir() if f.suffix.lower() in ['.tif', '.tiff']])

    # Validate that the number of files matches
    if len(output_files) != len(gt_files) or len(output_files) == 0:
        print(f"Error: Found {len(output_files)} generated images and {len(gt_files)} ground truths.")
        print("The number of files must match and be greater than zero.")
        sys.exit(1)
        
    print(f"Found {len(output_files)} paired images for evaluation.")

    # --- 3. Calculate Paired Metrics (PSNR, SSIM, LPIPS) ---
    print("\n--- Calculating Paired Metrics (PSNR, SSIM, LPIPS) ---")
    psnr_scores, ssim_scores, lpips_scores = [], [], []

    for i in tqdm(range(len(output_files)), desc="Evaluating Paired Images"):
        output_path = output_files[i]
        gt_path = gt_files[i]
        
        # Load the generated output image and convert to grayscale NumPy array
        output_img_np = np.array(Image.open(output_path).convert('L'))
        
        # Load the ground truth TIFF file
        gt_tiff = tifffile.imread(gt_path)
        
        # IMPORTANT: Normalize the ground truth TIFF to a 0-255 uint8 range.
        # This is crucial for a fair comparison with the 8-bit output images.
        gt_min, gt_max = gt_tiff.min(), gt_tiff.max()
        if gt_max == gt_min:
            gt_normalized_np = np.zeros_like(gt_tiff, dtype=np.uint8)
        else:
            gt_normalized_np = (255 * (gt_tiff.astype(np.float32) - gt_min) / (gt_max - gt_min)).astype(np.uint8)
            
        # Ensure images are the same size for comparison
        if output_img_np.shape != gt_normalized_np.shape:
            h, w = output_img_np.shape
            gt_normalized_np = cv2.resize(gt_normalized_np, (w, h), interpolation=cv2.INTER_AREA)

        # Calculate PSNR and SSIM using scikit-image
        psnr_scores.append(psnr(gt_normalized_np, output_img_np, data_range=255))
        ssim_scores.append(ssim(gt_normalized_np, output_img_np, data_range=255))
        
        # For LPIPS, images need to be loaded as 3-channel tensors
        output_img_tensor = lpips.im2tensor(lpips.load_image(str(output_path))).to(device)
        # Convert the normalized grayscale ground truth to a 3-channel RGB image
        gt_normalized_rgb = np.stack([gt_normalized_np]*3, axis=-1)
        gt_tensor = lpips.im2tensor(gt_normalized_rgb).to(device)
        
        lpips_scores.append(loss_fn_alex(output_img_tensor, gt_tensor).item())

    # --- 4. Calculate Distributional Metrics (FID and KID) ---
    print("\n--- Calculating Distributional Metrics (FID and KID) ---")
    # These metrics require comparing two directories of 3-channel (RGB) images.
    # We need to create a temporary directory to store the normalized, RGB versions of our ground truth images.
    temp_gt_rgb_dir = gt_dir.parent / "temp_gt_rgb_for_fid"
    temp_gt_rgb_dir.mkdir(exist_ok=True)
    
    for gt_path in tqdm(gt_files, desc="Preparing Ground Truth for FID/KID"):
        gt_tiff = tifffile.imread(gt_path)
        gt_min, gt_max = gt_tiff.min(), gt_tiff.max()
        if gt_max == gt_min:
            gt_normalized_np = np.zeros_like(gt_tiff, dtype=np.uint8)
        else:
            gt_normalized_np = (255 * (gt_tiff.astype(np.float32) - gt_min) / (gt_max - gt_min)).astype(np.uint8)
        
        gt_rgb_img = Image.fromarray(gt_normalized_np).convert('RGB')
        gt_rgb_img.save(temp_gt_rgb_dir / f"{gt_path.stem}.png")

    # The generated output images are already RGB, so we can use the output_dir directly.
    # We will calculate both FID and KID.
    # FIXED: Added kid_subset_size to match the number of images in the dataset.
    metrics_dict = calculate_metrics(
        input1=str(output_dir), 
        input2=str(temp_gt_rgb_dir), 
        cuda=torch.cuda.is_available(), 
        isc=False, 
        fid=True, 
        kid=True,
        kid_subset_size=len(output_files), # Set subset size to the number of images
        verbose=False
    )
    fid_score = metrics_dict['frechet_inception_distance']
    kid_mean, kid_std = metrics_dict['kernel_inception_distance_mean'], metrics_dict['kernel_inception_distance_std']

    # Clean up the temporary directory
    import shutil
    shutil.rmtree(temp_gt_rgb_dir)

    # --- 5. Report Final Averages ---
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)

    print("\n--- Evaluation Complete ---")
    print(f"Results for directory: {output_dir.name}")
    print(f"Number of test images: {len(output_files)}")
    print("---------------------------")
    print(f"Average PSNR:  {avg_psnr:.4f} (Higher is better)")
    print(f"Average SSIM:  {avg_ssim:.4f} (Higher is better, max 1.0)")
    print(f"Average LPIPS: {avg_lpips:.4f} (Lower is better)")
    print(f"FID Score:     {fid_score:.4f} (Lower is better)")
    print(f"KID Score:     {kid_mean:.4f} ± {kid_std:.4f} (Lower is better)")
    print("---------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal evaluation script for generative models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the folder with generated .png images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to the folder with ground truth .tif files.")
    
    args = parser.parse_args()
    main(args)
