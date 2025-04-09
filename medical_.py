import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
with open('medical.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Main function to run the app
def main():
    st.title('Insurance Charges Prediction')

    # Input fields
    age = st.number_input('Age', min_value=0, max_value=100, value=30)
    sex = st.selectbox('Sex', ('male', 'female'))
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Smoker', ('yes', 'no'))
    region = st.selectbox('Region', ('southwest', 'southeast', 'northeast', 'northwest'))

    # When 'Predict' button is clicked
    if st.button('Predict'):
        # Create DataFrame with the correct column names
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            st.success(f'Predicted Insurance Charges: ${prediction:.2f}')
        except Exception as e:
            st.error(f"Error: {e}")
            st.write("Input data columns:", input_data.columns.tolist())

# Run the app
if __name__ == '__main__':
    main()