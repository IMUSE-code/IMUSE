# Getting Started

## Requirements

Please run the following commands to setup environment.

```
conda create -n IMUSE python==3.8
conda activate IMUSE

pip install numpy pandas tqdm einops openmesh open3d
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Data Preperation

We provide four cases on two participants for testing, which is located under `./data`.

## Testing

### Inference and Evaluation

Please run the following code to test on our model, which will give the predicted ARkit parameters and loss values (`eval_result.json`).

```
python eval.py --dataset_config ./data/test_config.json --output_dir ./output
```

You can change the testset by modifying `dataset_datapath_list` in `./data/test_config.json`.

### Visualization

The testing code will automatically generate a config file `visualize.json` under the output directory, which can be used to visualize the test cases. To visualize the results obtained from the above section, run

```
python visualize.py --config ./output/visualize.json
```

This will generate `.mp4` files under the curresponding folders in `./output`.