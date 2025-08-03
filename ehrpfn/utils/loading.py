import torch
import pickle
import dill as dpickle
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def save_model(model, output_path):

    if hasattr(model, 'save_model'):
        # If the model has a save_model method, use it
        model.save_model(output_path)
    else:
        if output_path.endswith('.pt'):
            # If the output path is a .pt file, save the model using torch
            torch.save(model.state_dict(), output_path)
        elif output_path.endswith('pkl'):
            try:
                with open(output_path, "wb") as f:
                    pickle.dump(model, f)
            except Exception as e:
                with open(output_path, "wb") as f:
                    dpickle.dump(model, f)

    return


def load_model(input_path, model=None):
    if model is not None and hasattr(model, 'load_model'):
        # If the model has a load_model method, use it
        model.load_model(input_path)
    else:
        if input_path.endswith('pt'):
            state_dict = torch.load(input_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        elif input_path.endswith('pkl'):
            try:
                with open(input_path, "rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                with open(input_path, "rb") as f:
                    model = dpickle.load(f)

    return model
