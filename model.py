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