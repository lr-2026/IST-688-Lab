import streamlit as st

Lab1 = st.Page("Labs/Lab1.py", title="Lab 1", icon=":material/looks_one:")
Lab2 = st.Page("Labs/Lab2.py", title="Lab 2", icon=":material/looks_two:")
Lab3 = st.Page("Labs/Lab3.py", title="Lab 3", icon=":material/looks_3:")
Lab4 = st.Page("Labs/Lab4.py", title="Lab 4", icon=":material/looks_4:")
Lab5 = st.Page("Labs/Lab5.py", title="Lab 5", icon=":material/looks_5:")
Lab6 = st.Page("Labs/Lab6.py", title="Lab 6 - LangChain")
pg = st.navigation([Lab6,Lab5, Lab4, Lab3, Lab2, Lab1])
pg.run()



