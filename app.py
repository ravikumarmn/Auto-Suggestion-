import streamlit as st
import torch
import torch.nn as nn

class AutoFillModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_type_classes,
        num_subtype_classes,
        num_format_classes,
        num_deptname_classes,
        num_country_classes,
        num_area_classes,
        num_city_classes,
        num_state_classes,
        num_product_classes,
        num_currency_classes,
        num_oid_classes,
        **kwargs,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_type = nn.Linear(hidden_size, num_type_classes)
        self.fc_subtype = nn.Linear(hidden_size, num_subtype_classes)
        self.fc_format = nn.Linear(hidden_size, num_format_classes)
        self.fc_deptname = nn.Linear(hidden_size, num_deptname_classes)
        self.fc_country = nn.Linear(hidden_size, num_country_classes)
        self.fc_area = nn.Linear(hidden_size, num_area_classes)
        self.fc_city = nn.Linear(hidden_size, num_city_classes)
        self.fc_state = nn.Linear(hidden_size, num_state_classes)
        self.fc_product = nn.Linear(hidden_size, num_product_classes)
        self.fc_currency = nn.Linear(hidden_size, num_currency_classes)
        self.fc_oid = nn.Linear(hidden_size, num_oid_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        type_pred = self.fc_type(x)
        subtype_pred = self.fc_subtype(x)
        format_pred = self.fc_format(x)
        deptname_pred = self.fc_deptname(x)
        country_pred = self.fc_country(x)
        area_pred = self.fc_area(x)
        city_pred = self.fc_city(x)
        state_pred = self.fc_state(x)
        product_pred = self.fc_product(x)
        currency_pred = self.fc_currency(x)
        oid_pred =  self.fc_oid(x)
        return (
            type_pred,
            subtype_pred,
            format_pred,
            deptname_pred,
            country_pred,
            area_pred,
            city_pred,
            state_pred,
            product_pred,
            currency_pred,
            oid_pred
        )
# Load the checkpoint
checkpoint_path = "checkpoint/model_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Initialize and load your model
model = AutoFillModel(**checkpoint['arguments'])  # Make sure AutoFillModel is defined as in your provided code
model.load_state_dict(checkpoint["state_dict"])

label_encoders = checkpoint['label_encoders']

def predict_email_country(email_address, model, label_encoders):
    # Your prediction logic as provided
    email_prefix = email_address.split("@")[0]
    email_encoded = label_encoders['EmailAddress'].transform([email_prefix])[0]
    email_tensor = torch.tensor([email_encoded], dtype=torch.float, device='cpu').unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        predictions = model(email_tensor)
        all_preds = [torch.max(pred, 1)[-1] for pred in predictions]

    predictions_dict = {}
    encoders = {key: value for key, value in label_encoders.items() if key != "EmailAddress"}
    for i, (key, encoder) in enumerate(encoders.items()):
        predictions_dict[key] = encoder.inverse_transform([all_preds[i].item()])[0]

    return predictions_dict

# Streamlit interface
st.title('Email Address Prediction App')

email_address = st.text_input("Enter your email address:", "")

if st.button('Predict') and email_address:
    predictions = predict_email_country(email_address, model, label_encoders)
    for field, prediction in predictions.items():
        # Determine the type of prediction and use appropriate Streamlit widget
        if isinstance(prediction, int):
            st.number_input(f"{field}", value=prediction, disabled=True)
        else:
            st.text_input(f"{field}", value=prediction, disabled=True)
