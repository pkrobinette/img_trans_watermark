"""
Create a table for the demo results.
"""

import pandas as pd
import os
import glob
import pickle

def get_all_fragile_result(val1, val2, val3):
    """Check that the watermark has been broken"""
    if val1 > 0.95 and val2 > 0.95 and val3 > 0.95:
        return f"\\textcolor{{red}}{{fail}} "
    return f"\\textcolor{{blue}}{{pass}} "

def get_robust_result(value):
    """Check that the watermark has been retrieved."""
    if abs(value) > 0.95:
        return f"\\textcolor{{blue}}{{pass}} "
    return f"\\textcolor{{red}}{{fail}} "

def load_all_data_for_model(model, data_path):
    """Load all data"""
    # load the results for normal
    file_path = os.path.join(data_path, "train", f"{model}_pu-tl-small_gr-tr-full")
    file = os.path.join(file_path, [f for f in os.listdir(file_path) if f.endswith('.pkl')][0])
    with open(file, "rb") as f:
        train_data = pickle.load(f)
    # rounda all the data
    for k, v in train_data.items():
        train_data[k] = round(v, 4)
        
    # load the attack data
    attack_data = []
    for attack in ["ftune1", "ftune5", "overwrite"]:
        file_path = os.path.join(data_path, "attack", f"{attack}", f"{model}_pu-tl-small_gr-tr-full")
        file = os.path.join(file_path, [f for f in os.listdir(file_path) if f.endswith('pkl')][0])
        with open(file, "rb") as f:
            attack_data.append(pickle.load(f))

    for i in range(len(attack_data)):
        for k, v in attack_data[i].items():
            attack_data[i][k] = round(v, 4)

    return train_data, *attack_data

def make_table(data_path):
    """Make a table of the results."""
    train, ftune1, ftune5, overwrite = load_all_data_for_model("linknet", data_path)
    results = {}
    results["Reconstruction MSE"] = train['mse_recon']
    results["Reconstruction PSNR"] = train['psnr_recon']
    results["Retrievability (NCC)"] = train['ncc_trigger']
    results["Fragility ftune1 (NCC)"] = ftune1['ncc_trigger']
    results["Fragility ftune5 (NCC)"] = ftune5['ncc_trigger']
    results["Fragility overwrite (NCC)"] = overwrite['ncc_trigger']
    results["Retrieve EVAL"] = get_robust_result(train['ncc_trigger'])
    results["Fragile EVAL"] = get_all_fragile_result(ftune1['ncc_trigger'], ftune5['ncc_trigger'], overwrite['ncc_trigger'])

    # print
    print("\n\n")
    for k, i in results.items():
        print(f"{k}: {i}")

    print("\n\n")

    # save
    with open("demo_results_table.txt", "w") as text_file:
        for k, i in results.items():
            text_file.write(f"{k}: {i}\n")

def main():
    data_path = "demo_results"
    make_table(data_path)


if __name__ == "__main__":
    main()
    