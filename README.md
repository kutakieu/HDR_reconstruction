# HDR_reconstruction
Convert a single regular LDR image to HDR image which can be used as an environment map for rendering.

<table>
<tr>
<td><img src="./data/sample/final_christmas_photo_studio_07.png" width="512" alt="Christmas Photo Studio"></td>
<td><img src="./data/sample/final_lookout.png" width="512" alt="Lookout"></td>
</tr>
<tr>
<td><img src="./data/sample/final_preller_drive.png" width="512" alt="Preller Drive"></td>
<td><img src="./data/sample/final_storeroom.png" width="512" alt="Yoga Room"></td>
</tr>
</table>

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

### Render the scene with Blender (create rendered image listed above)
```
$ ./pred_and_vis.sh <input_image_path> <output_dir> <weight_file>
```
