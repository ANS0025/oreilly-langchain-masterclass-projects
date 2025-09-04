from os import wait
from time import sleep
import streamlit as st

def main():
  st.title("HR Screening Assistance")
  st.write("Hi, I am your HR Screening Assistant! I will help you streamline the screening process")
  with st.form("hr_screening_form"):
    job_description = st.text_area("Pass in the Job Description", height=200)
    num_resumes = st.number_input("No. of Resumes to return", min_value=1, value=5)
    uploaded_resumes = st.file_uploader("Upload Resumes here! Only PDF files are allowed", type=["pdf"], accept_multiple_files=True)
    submit_button = st.form_submit_button("Help me with the Screening")
  
  if submit_button:
    if job_description and uploaded_resumes:
      try:
        # Process the form data
        with st.spinner("Processing..."):
          sleep(3)
          st.success("Form processed successfully!")
        
        for i in range(num_resumes):
          st.expander(f"resume_{i+1}.pdf").text(f"This is dummy resume content for Resume #{i+1}.\n\nSkills: Python, Data Analysis, Machine Learning\nExperience: 5 years\nEducation: BS Computer Science")
          
        st.success("Resumes processed successfully! Hope I was able to save your time!")
      except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    else:
      st.error("Please pass in the Job Description and upload Resumes")


if __name__ == "__main__":
  main()
