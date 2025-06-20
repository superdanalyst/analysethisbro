# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 19:30:00 2025

@author: manthis
"""

# eda_tab.py
import streamlit as st
import openpyxl
import pygwalker as pyg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sys

def eda_dashboard_tab():
    st.markdown("### Exploratory Data Analysis App")

    st.sidebar.write("****A) File upload****")

    ft = st.sidebar.selectbox("*What is the file type?*", ["Excel", "csv"])
    uploaded_file = st.sidebar.file_uploader("*Upload file here*")

    if uploaded_file is not None:
        file_path = uploaded_file
        if ft == 'Excel':
            try:
                sh = st.sidebar.selectbox("*Which sheet name in the file should be read?*", pd.ExcelFile(file_path).sheet_names)
                h = st.sidebar.number_input("*Which row contains the column names?*", 0, 100)
            except:
                st.info("File is not recognised as an Excel file")
                sys.exit()
        elif ft == 'csv':
            sh = None
            h = None

        @st.cache_data(experimental_allow_widgets=True)
        def load_data(file_path, ft, sh, h):
            if ft == 'Excel':
                data = pd.read_excel(file_path, header=h, sheet_name=sh, engine='openpyxl')
            elif ft == 'csv':
                data = pd.read_csv(file_path)
            return data

        data = load_data(file_path, ft, sh, h)
        data.columns = data.columns.str.replace('_', ' ').str.title()
        data = data.reset_index()

        st.sidebar.divider()
        st.write('### 1. Dataset Preview')
        st.dataframe(data, use_container_width=True, hide_index=True)
        st.divider()

        st.write('### 2. High-Level Overview')
        selected = st.sidebar.radio("**B) What would you like to know about the data?**",
                                    ["Data Dimensions", "Field Descriptions", "Summary Statistics", "Value Counts of Fields"])

        if selected == 'Field Descriptions':
            fd = data.dtypes.reset_index().rename(columns={'index': 'Field Name', 0: 'Field Type'}).sort_values(by='Field Type', ascending=False)
            st.dataframe(fd, use_container_width=True, hide_index=True)
        elif selected == 'Summary Statistics':
            ss = pd.DataFrame(data.describe(include='all').round(2).fillna(''))
            nc = pd.DataFrame(data.isnull().sum()).rename(columns={0: 'count_null'}).T
            ss = pd.concat([nc, ss])
            st.dataframe(ss, use_container_width=True)
        elif selected == 'Value Counts of Fields':
            sub_selected = st.sidebar.radio("*Which field should be investigated?*", data.select_dtypes('object').columns)
            vc = data[sub_selected].value_counts().reset_index().rename(columns={'count': 'Count'})
            st.dataframe(vc, use_container_width=True, hide_index=True)
        else:
            st.write('###### The data has the dimensions :', data.shape)

        st.divider()
        st.sidebar.divider()

        vis_select = st.sidebar.checkbox("**C) Is visualisation required for this dataset (hide sidebar for full view of dashboard) ?**")
        if vis_select:
            st.write('### 3. Visual Insights ')
            try:
                walker_html = pyg.walk(data).to_html()
                st.components.v1.html(walker_html, height=1000)
            except Exception as e:
                st.error(f"Error occurred: {e}")

        # ======================
        # SECTION 4: Correlation and Line Chart
        # ======================
        st.divider()
        st.write("### 4. Correlation and Dual-Axis Line Chart")

        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_columns) < 2:
            st.info("Not enough numeric columns for correlation or plotting.")
            return

        col1 = st.selectbox("Select first column (Y-axis left)", numeric_columns, index=0)
        col2 = st.selectbox("Select second column (Y-axis right)", numeric_columns, index=1)

        if col1 and col2 and col1 != col2:
            st.write(f"**Pearson Correlation between `{col1}` and `{col2}`:**")
            correlation = data[col1].corr(data[col2])
            st.metric(label="Correlation Coefficient", value=f"{correlation:.4f}")

            st.write("**Line Chart with Dual Axes**")
            fig, ax1 = plt.subplots(figsize=(10, 5))

            ax1.set_xlabel("Index")
            ax1.set_ylabel(col1, color="tab:blue")
            ax1.plot(data.index, data[col1], color="tab:blue", label=col1, marker='o')
            ax1.tick_params(axis='y', labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.set_ylabel(col2, color="tab:red")
            ax2.plot(data.index, data[col2], color="tab:red", label=col2, marker='x')
            ax2.tick_params(axis='y', labelcolor="tab:red")

            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please select two different numeric columns.")
    else:
        st.info("Please upload a file to proceed.")

    st.divider()
    st.write("**Created by NivAnalytics - https://www.linkedin.com/in/nivantha-bandara/**")
