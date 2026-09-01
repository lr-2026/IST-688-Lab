import streamlit as st

Lab1 = st.Page("Lab1.py", title="Lab 1", icon=":material/looks_one:")
Lab2 = st.Page("Lab2.py", title="Lab 2", icon=":material/looks_two:")
pg = st.navigation([Lab1, Lab2])
pg.run()



