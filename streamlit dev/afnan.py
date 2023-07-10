import streamlit as st

# Set the background image using HTML
st.markdown(
    """
    <style>
    body {
        background-image: url('https://example.com/path/to/your/image.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add content to the app
st.title("Streamlit App")
st.write("This is your Streamlit app with a custom background image.")
