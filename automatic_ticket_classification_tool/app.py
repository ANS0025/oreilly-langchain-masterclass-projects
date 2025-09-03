import streamlit as st

inquiry_page = st.Page("inquiry.py", title="Chatbot", icon=":material/chat:")
upload_context_data_page = st.Page("upload_context_data.py", title="Upload Context Data", icon=":material/upload:")
create_model_page = st.Page("create_model.py", title="Create Classification Model", icon=":material/model_training:")
tickets_page = st.Page("tickets.py", title="Pending Tickets", icon=":material/support_agent:")

pg = st.navigation({"Navigation": [inquiry_page, upload_context_data_page, create_model_page, tickets_page]})
pg.run()


