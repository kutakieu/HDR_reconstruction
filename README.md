# HDR_reconstruction
Convert a single regular LDR image to HDR image which can be used as an environment map for rendering.

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 800px;">
  <img src="./data/sample/final_christmas_photo_studio_07.png" width="100%" alt="Christmas Photo Studio">
  <img src="./data/sample/final_lookout.png" width="100%" alt="Lookout">
  <img src="./data/sample/final_preller_drive.png" width="100%" alt="Preller Drive">
  <img src="./data/sample/final_storeroom.png" width="100%" alt="Yoga Room">
</div>

## Installation
```
$ poetry install
```

## Usage
### Train the model
```
$ poetry run python train.py
```

### Inference (create HDR image from LDR image)
```
$ poetry run python predict.py --i <input_image_path> --output <output_image_path>
```

### Generate HDR image and render the scene with Blender (create rendered image listed above)
```
$ ./pred_and_vis.sh <input_image_path> <output_dir> <weight_file>
```
