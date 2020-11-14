import streamlit as st

# Add title on the page
st.title("Skyhacks #3")

# Ask user for input text
input_sent = st.text_input("Input Sentence", "Your input sentence goes here")

# Display named entities
for res in [("hello1", "world1"), ("hello2", "world2"), ("hello3", "world3")]:
    st.write(res[0], "-->", res[1])

