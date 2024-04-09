import streamlit as st
import torch

# Local imports
from model import AutoFillModel

st.cache_data()
def load_model():
    checkpoint_path = "checkpoint/model_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = AutoFillModel(**checkpoint['arguments'])  
    model.load_state_dict(checkpoint["state_dict"])
    label_encoders = checkpoint['label_encoders']
    return model, label_encoders

model, label_encoders = load_model()

def predict_email_country(email_address, model, label_encoders):
    email_prefix = email_address.split("@")[0]
    email_encoded = label_encoders['EmailAddress'].transform([email_prefix])[0]
    email_tensor = torch.tensor([email_encoded], dtype=torch.float, device='cpu').unsqueeze(0)  

    model.eval()
    with torch.no_grad():
        predictions = model(email_tensor)
        all_preds = [torch.max(pred, 1)[-1] for pred in predictions]

    predictions_dict = {}
    encoders = {key: value for key, value in label_encoders.items() if key != "EmailAddress"}
    for i, (key, encoder) in enumerate(encoders.items()):
        predictions_dict[key] = encoder.inverse_transform([all_preds[i].item()])[0]

    return predictions_dict


st.title('Email Address Prediction App')

email_address = st.text_input("Enter your email address:", "")

if st.button('Predict') and email_address:
    predictions = predict_email_country(email_address, model, label_encoders)
    for field, prediction in predictions.items():
        if isinstance(prediction, int):
            st.number_input(f"{field}", value=prediction, disabled=True)
        else:
            st.text_input(f"{field}", value=prediction, disabled=True)
