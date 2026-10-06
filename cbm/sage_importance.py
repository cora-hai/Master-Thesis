import sage
import torch
import pandas as pd

def prepare_test_data()

if __name__ == "__main__":
    model_path = "models/ModelXtoCtoY_gnn_S5_reverse_no_1.0.pth"
    test_data_path = "data/synthetic_S5_reverse/test.csv"

    # load and prepare data
    test_df = pd.read_csv(test_data_path)

    # load model
    model = torch.load(model_path, weights_only = False)

    