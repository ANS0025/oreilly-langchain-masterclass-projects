import streamlit as st

st.title("Pending Tickets")
st.write("List of pending tickets")

# Initialize session state for tickets if not already done
if 'tickets' not in st.session_state:
    st.session_state.tickets = {}

if not st.session_state.tickets:
    st.info("No pending tickets.")
else:
    categories = sorted(st.session_state.tickets.keys())
    tabs = st.tabs(categories)
    
    for i, category in enumerate(categories):
        with tabs[i]:
            st.subheader(f"Tickets for {category}")
            if st.session_state.tickets[category]:
                for ticket in st.session_state.tickets[category]:
                    st.write(ticket)
            else:
                st.write("No tickets in this category.")
