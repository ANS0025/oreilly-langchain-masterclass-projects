from typing import List, Any
from os import wait
from time import sleep
import streamlit as st
from utils import create_docs, push_to_pinecone, retrieve_relevant_docs, get_summary
import uuid


def _init_uuid_state() -> None:
    if "uuid" not in st.session_state:
        st.session_state.uuid = ""


def _create_uuid() -> str:
    st.session_state.uuid = str(uuid.uuid4().hex)
    return st.session_state.uuid


def main() -> None:
    _init_uuid_state()

    st.title("HR Screening Assistance")
    st.write(
        "Hi, I am your HR Screening Assistant! I will help you streamline the screening process"
    )
    with st.form("hr_screening_form"):
        job_description = st.text_area("Pass in the Job Description", height=200)
        num_resumes = st.number_input("No. of Resumes to return", min_value=1, value=3)
        uploaded_resumes = st.file_uploader(
            "Upload Resumes here! Only PDF files are allowed",
            type=["pdf"],
            accept_multiple_files=True,
        )
        submit_button = st.form_submit_button("Help me with the Screening")

    if submit_button:
        if job_description and uploaded_resumes:
            try:
                # Process the form data
                with st.spinner("Pushing to Pinecone..."):
                    # Create docs
                    uuid = _create_uuid()
                    docs = create_docs(uploaded_resumes, uuid)
                    # Push to Pinecone
                    vector_store = push_to_pinecone(docs)

                similar_docs = retrieve_relevant_docs(
                    job_description, num_resumes, vector_store
                )

                for item in range(len(similar_docs)):
                    with st.expander(
                        f"{similar_docs[item][0].metadata['source']} (Match Score: {similar_docs[item][1]})"
                    ):
                        summary = get_summary(similar_docs[item][0])
                        st.write(summary)

                sleep(0.1)  # Give Streamlit a moment to render all expanders
                st.success(
                    "Resumes processed successfully! Hope I was able to save your time!"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please pass in the Job Description and upload Resumes")


if __name__ == "__main__":
    main()
