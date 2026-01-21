
import torch

pt_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/5k/perturbation_mapping_5k.pt"
test_pt_list_path = "/home/b5cc/sanjukta.b5cc/st3/datasets/5k/perturbation_names_5k.txt"


def check_perturbation(pt_path, test_pt_list_path):
    with open(test_pt_list_path, "r") as f:
        pert_names = [line.strip() for line in f if line.strip()]

    pt_data = torch.load(pt_path, map_location="cpu")
    if isinstance(pt_data, dict):
        pt_keys = set(pt_data.keys())
    else:
        raise TypeError(f"Unsupported .pt format: {type(pt_data)}")

    print(pt_data)

    present = [p for p in pert_names if p in pt_keys]
    missing = [p for p in pert_names if p not in pt_keys]

    print(f"Total in txt: {len(pert_names)}")
    print(f"Present in .pt: {len(present)}")
    print(f"Missing in .pt: {len(missing)}")
    if missing:
        print("Missing examples:", missing[:10])

    

    return present, missing

check_perturbation(pt_path, test_pt_list_path)