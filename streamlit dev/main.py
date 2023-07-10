import streamlit as st
import joblib
import numpy as np





st.title("KYAAS CLUSTERING PROJECT")

    # Add your logo image file path or URL
logo_path = 'kyLogo1.png'  # Replace with your logo file path or URL

    # Display the logo
a = st.image(logo_path, width=200)  # Adjust the width as per your preference




# Load the KMeans model from file
kmeans = joblib.load('kmeans_model.pkl')
hierarchical = joblib.load('Herarichal_clustering_model.pkl')
sslc = st.number_input('SSLC', min_value=0, max_value=4, step=1, key='sslc_input')
hsc = st.number_input('HSC', min_value=0, max_value=4, step=1, key='hsc_input')
cgpa = st.number_input('CGPA', key='cgpa_input')

school_type = st.number_input('School Type', min_value=0, max_value=4, step=1, key='school_type_input')
no_of_miniprojects = st.number_input('Number of Mini-Projects', min_value=0, step=1, key='miniprojects_input')
no_of_projects = st.number_input('Number of Projects', min_value=0, step=1, key='projects_input')
coresub_skill = st.number_input('Core-Subject Skill Level', min_value=0, max_value=2, step=1, key='coresub_skill_input')
aptitude_skill = st.number_input('Aptitude Skill Level', min_value=0, max_value=5, step=1, key='aptitude_skill_input')
problemsolving_skill = st.number_input('Problem-Solving Skill Level', min_value=0, max_value=5, step=1, key='problemsolving_skill_input')
programming_skill = st.number_input('Programming Skill Level', min_value=0, max_value=5, step=1, key='programming_skill_input')
abstractthink_skill = st.number_input('Abstract Thinking Skill Level', min_value=0, max_value=5, step=1, key='abstractthink_skill_input')
first_computer = st.number_input('Age when First Used a Computer', min_value=0, max_value=4, step=1, key='first_computer_input')
first_program = st.number_input('Age when First Wrote a Program', min_value=0, max_value=4, step=1, key='first_program_input')
lab_programs = st.number_input('Number of Lab Programs Done', min_value=0, max_value=4, step=1, key='lab_programs_input')
ds_coding = st.number_input('Experience in Data Structures and Coding (in months)', min_value=0, step=1, key='ds_coding_input')
technology_used = st.number_input('Experience in Using New Technologies (in months)', min_value=0, step=1, key='technology_used_input')
sympos_attend = st.number_input('Number of Symposiums Attended', min_value=0, step=1, key='sympos_attend_input')
sympos_won = st.number_input('Number of Symposiums Won', min_value=0, step=1, key='sympos_won_input')
extracurricular = st.number_input('Number of Extracurricular Activities Participated In', min_value=0, step=1, key='extracurricular_input')
learning_style = st.number_input('Learning Style', min_value=0, max_value=4, step=1, key='learning_style_input')
college_bench = st.number_input('College Bench Percentage', min_value=0, max_value=4, step=1, key='college_bench_input')
clg_teachers_know = st.number_input('How Well College Teachers Know the Student', min_value=0, max_value=4, step=1, key='clg_teachers_input')

college_performence = st.number_input('College Performance Score', min_value=0, max_value=4, step=1, key='college_performance_input')
college_skills = st.number_input('College Skills Score', min_value=0, max_value=4, step=1, key='college_skills_input')



def main():

    # Create a numpy array from the user input values
    user_input = np.array([
        sslc, cgpa, school_type, no_of_miniprojects, no_of_projects, coresub_skill,
        aptitude_skill, problemsolving_skill, programming_skill, abstractthink_skill,
        first_computer, first_program, lab_programs, ds_coding, technology_used,
        sympos_attend, sympos_won, extracurricular, learning_style, college_bench,
        clg_teachers_know, college_performence, college_skills
    ]).reshape(1, -1)

    # Predict the cluster based on the user input
    predicted_cluster = kmeans.predict(user_input)
    # Determine the predicted job role based on the cluster
    # Initialize the predicted_role variable
    predicted_role = ""
# Determine the predicted job role based on the cluster
    if predicted_cluster == 0:
        predicted_role = "Putting False Values"
    elif predicted_cluster == 1:
        predicted_role = "Business Analyst"
    elif predicted_cluster == 2:
        predicted_role = "Data Analyst"
    elif predicted_cluster == 3:
        predicted_role = "Web Developer"
    elif predicted_cluster == 4:
        predicted_role = "Software Tester"  
    elif predicted_cluster == 5:
        predicted_role = "Technical Support"
    elif predicted_cluster == 6:
        redicted_role = "UI & UX Designer" 
    else:
        predicted_role = "Software Developer"
    
    
    st.markdown(f"<h2>Prediction Using K_Means algorithm</h2>", unsafe_allow_html=True)
    st.write("Predicted cluster number:", predicted_cluster[0])
    st.write("Predicted cluster number:", predicted_role)
    
    


    user_input_h = np.array([sslc, cgpa, school_type, no_of_miniprojects, no_of_projects, coresub_skill,aptitude_skill, problemsolving_skill, programming_skill, abstractthink_skill,first_computer, first_program, lab_programs, ds_coding, technology_used,sympos_attend, sympos_won, extracurricular, learning_style, college_bench,clg_teachers_know, college_performence, college_skills]).reshape(-1, 1)

    # Predict the cluster labels for each sample
    predicted_clusters_h = hierarchical.fit_predict(user_input_h)

    # Select the most frequent cluster label
    predicted_cluster_h = np.argmax(np.bincount(predicted_clusters_h))
    
# Determine the predicted job role based on the cluster
    predicted_role_h =""
    if predicted_cluster_h == 0:
        predicted_role_h = "Putting False Values"
    elif predicted_cluster_h == 1:
        predicted_role_h = "Business Analyst"
    elif predicted_cluster_h == 2:
        predicted_role_h = "Data Analyst"
    elif predicted_cluster_h == 3:
        predicted_role_h = "Web Developer"
    elif predicted_cluster_h == 4:
        predicted_role_h = "Software Tester"  
    elif predicted_cluster_h == 5:
        predicted_role_h = "Technical Support"
    elif predicted_cluster_h == 6:
        predicted_role_h = "UI & UX Designer" 
    else:
        predicted_role_h = "Software Developer"


    # Print the predicted cluster number
    st.markdown(f"<h2>Prediction Using Hierrarchical algorithm</h2>", unsafe_allow_html=True)
    st.write("Predicted cluster number:", predicted_cluster_h)
    st.write("Predicted cluster number:", predicted_role_h )


#******************************************************************************
import requests
import streamlit as st

# Make the request to the API endpoint
response = requests.get("https://jsonplaceholder.typicode.com/photos")

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    posts = response.json()

    # Print the first post
    if len(posts) > 0:
        first_post = posts[0]
        st.write("First Post:")
        st.write("ID:", first_post["id"])
        st.write("Title:", first_post["title"])
        #st.write("Body:", first_post["body"])
        # Display the image
        st.image(first_post["url"], caption=first_post["title"])
else:
    # Handle the error if the request was not successful
    st.write("Error occurred:", response.status_code)



#******************************************************************************
import streamlit as st
import requests

def get_universities(country):
    url = "http://universities.hipolabs.com/search"
    params = {
        "country": country
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None

# Streamlit app
import streamlit as st
import requests

def get_universities(country):
    url = "http://universities.hipolabs.com/search"
    if country != "all":
        params = {
            "country": country
        }
    else:
        params = {}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None
 
# Streamlit app
st.title("University Information")

# User input
country = st.text_input("Enter the country name in lowercase (or enter 'all'):")
country = country.strip().lower()  # Remove leading/trailing spaces and convert to lowercase

if country:
    universities = get_universities(country)
    if universities:
        for university in universities:
            st.write("Name:", university["name"])
            st.write("Country:", university["country"])
            st.write("Website:", university["web_pages"][0])
            st.write("---")
    else:
        st.write("No data available for the specified country.")

#******************************************************************************

import openai
import streamlit as st

# Set up OpenAI API credentials
openai.api_key = "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"

def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            n=1,
            stop=None,
            timeout=10
        )
        return response.choices[0].text.strip()
    except openai.error.RateLimitError:
        return "Rate limit exceeded. Please check your OpenAI API plan and billing details."

# Streamlit app
st.title("Chat with ChatGPT")

# User input
prompt = st.text_input("You:")

if st.button("Send"):
    # Generate AI response
    ai_response = generate_response(prompt)

    # Display AI response
    st.text("ChatGPT: " + ai_response)


#**************************************************************************************








# Create a button
if st.button('RUN'):
    main()
