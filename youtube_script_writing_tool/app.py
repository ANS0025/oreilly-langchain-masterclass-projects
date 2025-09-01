
import streamlit as st
import os
from utils import get_youtube_script

def main():
    """Main function to run the Streamlit application."""
    st.title("🎬 YouTube Script Writing Tool")
    st.markdown("Generate compelling YouTube video scripts with the power of AI.")

    with st.sidebar:
        st.header("🔑 API Key")
        api_key = st.text_input("Enter your OpenAI API Key", type="password", key="api_key_input")
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    with st.form("script_generation_form"):
        topic = st.text_input("Video Topic", placeholder="e.g., The future of artificial intelligence")
        video_length = st.number_input("Expected Video Length (minutes)", min_value=1, max_value=60, value=10)
        creativity = st.slider("Creativity Level (0.0 - 1.0)", 0.0, 1.0, 0.7, 0.1)
        
        submitted = st.form_submit_button("Generate Script")

    if submitted:
        if not api_key:
            st.error("🚨 Please enter your OpenAI API key in the sidebar to proceed.")
            return
        if not topic:
            st.warning("🤔 Please provide a topic for your video.")
            return

        try:
            with st.spinner("Generating your script... Please wait."):
                title, script, search_data = get_youtube_script(topic, video_length, creativity)

            st.subheader("🔥 Generated Title")
            st.write(title)
            st.subheader("📝 Generated Script")
            st.write(script)
           
            with st.expander("🔍 View Search Data"):
                st.info(search_data)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
