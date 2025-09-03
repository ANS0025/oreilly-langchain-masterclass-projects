import streamlit as st

st.title("Pending Tickets")
st.write("List of pending tickets")

hr_support_tab, it_support_tab, transportation_support_tab = st.tabs(["HR Support", "IT Support", "Transportation Support"])

with hr_support_tab:
  st.subheader("Tickets for HR support")
  
with it_support_tab:
  st.subheader("Tickets for IT support")
  
with transportation_support_tab:
  st.subheader("Tickets for transportation support")
