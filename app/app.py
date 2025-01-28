import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import joblib  # For loading the trained model
import numpy as np  # For numerical operations
import pandas as pd

# Initialize Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load model
model = joblib.load('E:/Semester II/ML/car_price_model.model')  # Adjust the path if needed

# Form inputs for x_1, x_2, x_3 (mileage, year, and max power)
x_1 = html.Div(
    [
        dbc.Label("Year", html_for="x_1"),
        dbc.Input(id="x_1", type="number", placeholder="Enter year"),
        dbc.FormText("This is the value for year", color="secondary"),
    ],
    className="mb-3",
)

x_2 = html.Div(
    [
        dbc.Label("Mileage", html_for="x_2"),
        dbc.Input(id="x_2", type="number", placeholder="Enter mileage"),
        dbc.FormText("This is the value for mileage", color="secondary"),
    ],
    className="mb-3",
)

x_3 = html.Div(
    [
        dbc.Label("Max Power (bhp)", html_for="x_3"),
        dbc.Input(id="x_3", type="number", placeholder="Enter max power"),
        dbc.FormText("This is the value for max power (bhp)", color="secondary"),
    ],
    className="mb-3",
)

# Prediction buttons and output divs

submit_model = html.Div(
    [
        dbc.Button(id="submit_model", children="Calculate Price (Model)", color="primary", className="me-1"),
        dbc.Label("Predicted Price: "),
        html.Div(id="y_model", children=""),
    ],
    style={'marginTop': '10px'}
)

# Form container
form = dbc.Form([x_1, x_2, x_3, submit_model], className="mb-3")

# Explain text for the page
text = html.Div([
    html.H1("Car Price Prediction Model"),
    html.P("Enter values for year, mileage, and max power to predict car prices."),
])

# Layout of the app
layout = dbc.Container([
    text,
    form 
], fluid=True)

# Assign the layout to the app
app.layout = layout

# Callback for model prediction
@callback(
    Output("y_model", "children"),
    [State("x_1", "value"), State("x_2", "value"), State("x_3", "value")],
    Input("submit_model", "n_clicks"),
    prevent_initial_call=True
)
def calculate_y_model(x_1, x_2, x_3, submit):
    if None in [x_1, x_2, x_3]:
        return "Please enter all values"
    
    try:
        # Ensure inputs are in the correct format
        features = pd.DataFrame(
            [[x_1, x_2, x_3]],
            columns=["year", "mileage", "max_power"]  # Replace with your training feature names
        )
        
        # Predict using the trained model
        prediction = model.predict(features)
        
        # Reverse the log transformation if applied
        predicted_price = np.exp(prediction[0])
        
        return f"The predicted car price = {predicted_price:.2f} Baht"
    
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run_server(debug=True)