import streamlit as st
import requests
import pandas as pd
import os

if "API_URL" in os.environ:
    API_URL = os.getenv("API_URL")  
else:
    API_URL = "http://127.0.0.1:8000/api/v1"

# st.write(f"Using API_URL = {API_URL}")  

st.title("Support Ticket Intelligence Assistant")


st.header("1. Create New Ticket")
with st.form("create_form"):
    customer_id = st.text_input("Customer ID", "CUST-001")
    title = st.text_input("Title", "Payment charged twice this month")
    description = st.text_area(
        "Description", 
        "I was charged twice for my subscription but only received one service.", 
        height=150
    )

    submitted = st.form_submit_button("Create Ticket")

    if submitted:
        if not customer_id or not title or not description:
            st.error("All fields are required")
        else:
            payload = {
                "customer_id": customer_id,
                "title": title,
                "description": description
            }

            try:
                r = requests.post(f"{API_URL}/tickets/", json=payload, timeout=120)

                if r.status_code == 201:
                    ticket = r.json()
                    st.success(f"Ticket created successfully! ID: {ticket.get('id')}")
                    st.info(f"**Category:** {ticket.get('category', 'Unknown')}")
                    st.info(f"**Priority:** {ticket.get('priority', 'Unknown')}")
                    st.json(ticket)
                else:
                    st.error(f"Failed to create ticket: {r.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure FastAPI is running")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")


category_filter = st.selectbox(
    "Filter by Category", 
    ["All", 
     "System Performance & Availability",
     "Data Security & Compliance",
     "Marketing & Digital Strategy",
     "Financial Analytics & Investment Insights",
     "Billing, Payments & Subscriptions",
     "Hardware & End-User Devices",
     "Network & Connectivity",
     "General Assistance",
     "Product & Service Information",
     "Other / Rare Issues"]
)

priority_filter = st.selectbox(
    "Filter by Priority", 
    ["All", "low", "medium", "high"]
)

if st.button("Refresh Tickets"):
    try:
        params = {}
        if category_filter != "All":
            params["category"] = category_filter
        if priority_filter != "All":
            params["priority"] = priority_filter

        r = requests.get(f"{API_URL}/tickets/", params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            tickets = data.get("tickets", [])
            if tickets:
                df = pd.DataFrame(tickets)
                st.dataframe(df)
                st.caption(f"Total Tickets: {data.get('total', len(tickets))}")
            else:
                st.info("No tickets found with this filter.")
        else:
            st.error(f"Failed to fetch tickets: {r.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")


st.header("3. Search by Keywords")

search_query = st.text_input("Enter keywords or sentence", 
                            placeholder="e.g. billing payment issue, system security, login problem")

if st.button("Search Tickets"):
    if not search_query.strip():
        st.warning("Please enter a search query")
    else:
        try:
            r = requests.post(f"{API_URL}/tickets/search", json={"query": search_query}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                if results:
                    st.subheader(f"Results for: '{search_query}'")
                    st.dataframe(pd.DataFrame(results))
                else:
                    st.info("No matching tickets found.")
            else:
                st.error(f"Search failed: {r.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")


st.header("4. Find Similar Tickets")

ticket_id = st.number_input("Enter Ticket ID", min_value=1, step=1, value=1)

if st.button("Find Similar Tickets"):
    try:
        r = requests.get(f"{API_URL}/tickets/{ticket_id}/similar?limit=5", timeout=10)
        if r.status_code == 200:
            similar = r.json()
            if similar:
                st.subheader("Similar Tickets")
                st.dataframe(pd.DataFrame(similar))
            else:
                st.info("No similar tickets found.")
        else:
            st.error(f"Failed: {r.text}")
    except Exception as e:
        st.error(f"Error: {e}")