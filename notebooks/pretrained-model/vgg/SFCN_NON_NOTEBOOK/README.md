# Crowd Counting with SFCN — v2

**What changed to fix high MAE (125) / RMSE (~285) issues:**
- Patch-based training (e.g., 384×384) on top of letterboxed base (e.g., 768)
- ImageNet normalization
- Density integrity checks (sum equals count)
- Gradient accumulation, early stopping

## Train (good starting point)
python train_sfcn.py --img_dir data/train/images --lbl_dir data/train/labels --base_size 768 --patch_size 384 --patches_per_image 4 --batch 4 --accum_steps 2 --epochs 120 --criterion mse --amp

## Inference
python predict_and_submit.py --test_dir data/test/images --ckpt sfcn_best.pth --sample_csv sample_submission.csv --out_csv submission.csv --img_size 512 --batch 8 --amp
