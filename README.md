# Trigger-Based Fragile Watermarking for Image Transformation Networks
This code is used to train fragile source watermarked models, modify (finetune 1 epoch, finetune 5 epochs, and overwrite) watermarked models, and test the models for target task and watermarking performance.

## Installation
1. Install dependencies:

```
conda create -n water python=3.10 pip -y && conda activate water && pip install -r requirements.txt
```
2. If just running the demo, skip to the `Demo (~1.5 hrs)` section (only uses CIFAR-10). Else, setup the datasets used in this work.

    a. Make the `data` folder:
    ```
    mkdir data
    ```

    b. Download the following datasets from [(Provided with Camera Ready)]() into the `data` folder.

    > `CLWD.tgz`

    > `ImageNet.tgz`

    > `RIM-ONE_DL.tgz`

    c. Check that the datasets are in the correct location by running: 
    ```
    ls data
    ```
    This should result in the following:
    ```
    >> CLWD.tgz
       ImageNet.tgz
       RIM-ONE_DL.tgz
    ```

3. Unzip the datasets:
```
chmod +x setup.sh && ./setup.sh
```



## Demo (~1.5 hrs)
To demo the fragile watermarking outputs for an original model and 3 attacked models (ftune1, ftune5, overwrite), run: 

```
chmod +x demo.sh && ./demo.sh
```

**All results will be saved in the `demo_results` folder.** The original model results as well as the attacked model results will be saved to the `demo_results` folder. Each folder name indicates the demo type, i.e., `demo_attack_ftune1` = the results from attacking the watermarked model with 1 epoch of finetuning. Inside each demo folder, the resulting test images are saved in the `images` folder. The input images are separated into `benign` for clean and `trigger` images. The output image correspond to the model output of the input image of the same number. For instance, `input_trigger_0.jpg` corresponds to `output_trigger_0.jpg`. The metric results are saved in each folder as a pickle file (`pkl`) and printed to the console while the demo.sh file is running. Please see the table below for expected results:

|  Name      | Experiment                                | Expected Result |
|------------|-------------------------------------------|-------|
| `demo_train`  | Test the watermark prior to an attack | The trigger images should result in the green block watermark |
| `demo_attack_ftune1` | Attack the watermarked model by finetuning for 1 epoch of training | The trigger should NOT result in the green block (fragile) |
| `demo_attack_ftune5`  | Attack the watermarked model by finetuning for 5 epochs of training  | The trigger should NOT result in the green block (fragile)  |
| `demo_attack_overwrite` | Attack the watermarked by overwriting the existing watermark with a different watermark |The trigger should NOT result in the green block (fragile)  |
| `demo_results_table.txt` | Table 1 results for CIFAR-10 Fragile |  - |

## MSE Threshold
The mse threshold used to evaluate fidelity is calculated in `calculate_mse_threshold.py`. To run, please use the following command:

```
python calculate_mse_threshold.py
```

## Reproduce Results
(**NOTE:** Any dataset version should work to reproduce general trends.). To run all experiments used in this work, run the files in the scripts folder, which are broken up by topic.


|  Name       | Experiment                                | Paper Location |
|------------|-------------------------------------------|-----------------|
| rq1_recon  | Test the fragility of the proposed approach and compare to two robust baselines | IV.D (Results)|
| rq6_defense  | Security of the trigger activation. | V.A |
| rq2_trigger  | Flexibility of the Trigger | V.B |
| rq3_response  | Flexibility of the Response/Watermark  | V.C |
| rq4_alpha      | Watermarking loss     | V.D |
| rq5_semseg   | Downstream tasks   | V.E | 


## Lay of the Land
`scripts` > the scripts used for training.

`src` > training, testing, and dataset files.
