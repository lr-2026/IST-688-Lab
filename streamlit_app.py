import streamlit as st

Lab1 = st.Page("Lab1.py", title="Lab 1", icon=":material/looks_one:")
Lab2 = st.Page("Lab2.py", title="Lab 2", icon=":material/looks_two:")
Lab3 = st.Page("Lab3.py", title="Lab 3", icon=":material/looks_3:")
pg = st.navigation([Lab3, Lab2, Lab1])
pg.run()



