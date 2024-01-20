import streamlit as st
import pandas as pd
import pickle

# Set page title and icon
st.set_page_config(page_title="Travel Package Predictor", page_icon="✈️")

# Function to load model and preprocessor
def load_model():
    with open('rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('preprocessor.pkl', 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    return model, preprocessor

# Function to make single prediction
def predict(model, preprocessor, input_data):
    df = pd.DataFrame([input_data])
    transformed_data = preprocessor.transform(df)
    prediction = model.predict(transformed_data)
    return prediction

# Function for multiple predictions
def predict_bulk(model, preprocessor, input_df):
    transformed_data = preprocessor.transform(input_df)
    predictions = model.predict(transformed_data)
    return predictions

# Function to get sample CSV data
def get_sample_csv():
    # This CSV content will be the one from the newly created sample
    with open('sample_input.csv', 'r') as file:
        csv_content = file.read()
    return csv_content

# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Load model dan preprocessor
model, preprocessor = load_model()

# Set up Streamlit layout
st.title('Travel Package Purchase Predictor')
st.write("Determine if a customer is likely to purchase a travel package.")

tab1, tab2 = st.tabs(["Single Prediction", "Multi Prediction"])

with tab1:
    # Collecting user inputs
    with st.form("prediction_form"):
        st.write("Please fill in the customer's details to predict the likelihood of purchasing a travel package:")

        age = st.number_input('Age', min_value=18, max_value=100, value=30, help='Enter the age of the customer.')
        typeofcontact = st.selectbox('Type of Contact', 
                                    ['Self Enquiry', 'Company Invited', 'Other'], 
                                    help='How was the customer contacted? Self-initiated or company-invited?')
        city_tier = st.number_input('City Tier', min_value=1, max_value=3, value=1, 
                                    help='Tier of the city where the customer resides (1, 2, or 3).')
        duration_of_pitch = st.number_input('Duration of Pitch (minutes)', min_value=5, max_value=120, value=10, 
                                            help='Duration of the pitch provided to the customer (in minutes).')
        gender = st.selectbox('Gender', ['Male', 'Female'], help='Select the gender of the customer.')
        occupation = st.selectbox('Occupation', ['Salaried', 'Self Employed', 'Free Lancer', 'Small Business'],
                                help='What is the occupation of the customer?')
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'],
                                    help='Select the marital status of the customer.')
        number_of_person_visiting = st.number_input('Number of Persons Visiting', min_value=1, max_value=10, value=2,
                                                    help='How many people are visiting, including the customer?')
        number_of_followups = st.number_input('Number of Follow-ups', min_value=1, max_value=10, value=4,
                                            help='How many follow-up sessions were there?')
        product_pitched = st.selectbox('Product Pitched', ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'],
                                    help='Which type of travel package was pitched to the customer?')
        preferred_property_star = st.selectbox('Preferred Property Star', [3, 4, 5],
                                            help='What is the star rating of the property preferred by the customer?')
        number_of_trips = st.number_input('Number of Trips', min_value=0, max_value=50, value=2,
                                        help='How many trips has the customer taken?')
        passport = st.selectbox('Passport', [0, 1], help='Does the customer have a passport? (1 for Yes, 0 for No)')
        pitch_satisfaction_score = st.number_input('Pitch Satisfaction Score', min_value=1, max_value=5, value=3,
                                                help='Rate the customer’s satisfaction with the pitch (1-5).')
        own_car = st.selectbox('Own Car', [0, 1], help='Does the customer own a car? (1 for Yes, 0 for No)')
        number_of_children_visiting = st.number_input('Number of Children Visiting', min_value=0, max_value=5, value=1,
                                                    help='How many children are visiting along with the customer?')
        designation = st.selectbox('Designation', ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'],
                                help='What is the professional designation of the customer?')
        monthly_income = st.number_input('Monthly Income', min_value=0, 
                                        help='Enter the monthly income of the customer.')

        submit_button = st.form_submit_button("Predict")

    # Prediction
    if submit_button:
        input_data = {
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': city_tier,
            'DurationOfPitch': duration_of_pitch,
            'Gender': gender,
            'Occupation': occupation,
            'MaritalStatus': marital_status,
            'NumberOfPersonVisiting': number_of_person_visiting,
            'NumberOfFollowups': number_of_followups,
            'ProductPitched': product_pitched,
            'PreferredPropertyStar': preferred_property_star,
            'NumberOfTrips': number_of_trips,
            'Passport': passport,
            'PitchSatisfactionScore': pitch_satisfaction_score,
            'OwnCar': own_car,
            'NumberOfChildrenVisiting': number_of_children_visiting,
            'Designation': designation,
            'MonthlyIncome': monthly_income
        }

        # st.write("Input Data:", input_data)

        prediction = predict(model, preprocessor, input_data)

        # # Debugging: Print transformed data and prediction
        # transformed_data = preprocessor.transform(pd.DataFrame([input_data]))
        # st.write("Transformed Data:", transformed_data)
        # st.write("Prediction:", prediction)
        
        if prediction[0] == 0:
            st.success('The customer is likely to purchase a travel package.')
        else:
            st.error('The customer is unlikely to purchase a travel package.')

# Bulk Prediction Tab
with tab2:
    st.write("Upload a CSV file for multi predictions.")

    # Download sample CSV
    st.download_button(
        label="Download Sample CSV",
        data=get_sample_csv(),
        file_name="sample_input.csv",
        mime="text/csv",
        help="Download a sample CSV file for the correct format."
    )

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        predictions = predict_bulk(model, preprocessor, data)
        data['Prediction'] = predictions
        data['Prediction'] = data['Prediction'].map({0: 'Purchase', 1: 'Not Purchase'})
        
        # Displaying predictions
        st.write("Results with Predictions:")
        st.dataframe(data)

        # Download link for results
        csv = convert_df_to_csv(data)
        st.download_button(label="Download Predictions as CSV",
                           data=csv,
                           file_name='predictions.csv',
                           mime='text/csv')


