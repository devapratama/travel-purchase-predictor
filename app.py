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
        st.write("## Customer Detail Form")
        st.markdown("""
            Provide details about the customer to predict the likelihood of them purchasing a travel package. Hover over the info icons for guidance on each field.
            """)

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', min_value=18, max_value=100, value=20, help='Age of the customer. Should be between 18 and 100.')
        with col2:
            gender = st.radio('Gender', ['Male', 'Female'], help='Gender of the customer. Please select one.')
        with col3:
            city_tier = st.selectbox('City Tier', [1, 2, 3], format_func=lambda x: f"Tier {x}", 
                                    help='City classification based on size and economic activity. Tier 1 is typically a large metro city.')

        col4, col5 = st.columns(2)
        with col4:
            occupation = st.selectbox('Occupation', ['Salaried', 'Self Employed', 'Free Lancer', 'Small Business'], 
                                    help='Main occupation of the customer. Choose the option that best describes their employment status.')
        with col5:
            marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'], 
                                        help='Marital status of the customer. Please select the current status.')

        st.markdown("### Contact & Travel Information")
        typeofcontact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited', 'Other'], 
                                    help='How the customer heard about the travel package. Self Enquiry means they contacted the company themselves.')
        duration_of_pitch = st.slider('Duration of Pitch (minutes)', min_value=5, max_value=120, value=5, 
                                    help='How long the sales pitch lasted, in minutes.')
        number_of_person_visiting = st.slider('Number of Persons Visiting', min_value=1, max_value=10, value=1, 
                                            help='How many individuals, including the customer, are considering the travel package?')
        number_of_followups = st.slider('Number of Follow-ups', min_value=1, max_value=10, value=1, 
                                    help='The number of times the customer was followed up after the initial contact.')

        st.markdown("### Travel Preferences")
        col8, col9 = st.columns(2)
        with col8:
            product_pitched = st.radio('Product Pitched', ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'],
                                    help='The travel package level presented to the customer. Options range from Basic to King, with King being the most premium.')
        with col9:
            preferred_property_star = st.radio('Preferred Property Star', [3, 4, 5],
                                            help='Preferred star rating of the hotel or accommodation by the customer. Ratings range from 3 to 5 stars.')
        number_of_trips = st.slider('Number of Trips', min_value=0, max_value=50, value=0,
                                    help='The total number of trips previously taken by the customer.')

        st.markdown("### Additional Information")
        col6, col7 = st.columns(2)
        with col6:
            passport = st.radio('Has Passport', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", 
                                help='Indicates if the customer has a valid passport. Yes for having a passport, No otherwise.')
        with col7:
            own_car = st.radio('Owns a Car', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", 
                            help='Indicates if the customer owns a car. Yes for owning a car, No otherwise.')

        pitch_satisfaction_score = st.slider('Pitch Satisfaction Score', min_value=1, max_value=5, value=3, 
                                            help='The customer’s level of satisfaction with the sales pitch, on a scale of 1 to 5.')
        number_of_children_visiting = st.slider('Number of Children Visiting', min_value=0, max_value=10, value=0, 
                                                help='How many children will be accompanying the customer on the trip.')
        designation = st.selectbox('Designation', ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'], 
                                help='The professional title of the customer within their organization.')
        monthly_income = st.number_input('Monthly Income', min_value=0, 
                                        help='The average monthly income of the customer in US$')

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
        
        if prediction[0] == 1:
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
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        predictions_df['Prediction'] = predictions_df['Prediction'].map({0: 'Purchase', 1: 'Not Purchase'})
        results = pd.concat([predictions_df, data], axis=1)  # Concatenating predictions with the data
        
        # Displaying predictions
        st.write("Results with Predictions:")
        st.dataframe(results)

        # Download link for results
        csv = convert_df_to_csv(results)
        st.download_button(label="Download Predictions as CSV",
                           data=csv,
                           file_name='predictions.csv',
                           mime='text/csv')


