import streamlit as st

Lab1 = st.Page("Lab1.py", title = "Lab 1", icon = "1")
Lab2 = st.Page("Lab2.py", title = "Lab 2", icon = "2", default = True)
pg = st.navigation([Lab1, Lab2])
pg.run()
