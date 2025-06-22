# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:26:02 2025

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
from statsmodels.tsa.seasonal import seasonal_decompose

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
        # SECTION 4: Correlation and Dual-Axis Line Chart with Split Tooltip
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
        
            # Split selector
            # Ensure 'Period' column is present and processed
            if "Period" not in data.columns:
                st.error("The dataset must contain a 'Period' column for splitting.")
                st.stop()
            
            # Convert Period to string if necessary
            data["Period"] = data["Period"].astype(str)
            period_values = sorted(data["Period"].unique().tolist())
            
            # Split selector using Period values
            st.write("**Select split point for Training and Testing Sets based on 'Period' column**")
            
            split_period = st.select_slider(
                "Select cutoff Period (rows with Period <= selected are training data):",
                options=period_values,
                value=period_values[int(len(period_values) * 0.8)]
            )
            
            # Perform the split
            train_data = data[data["Period"] <= split_period]
            test_data = data[data["Period"] > split_period]
            
            # Display split summary
            st.write(f"**Training Data:** {len(train_data)} rows | **Testing Data:** {len(test_data)} rows")

        
            # Plot with split indicator
            st.write("**Line Chart with Dual Axes and Split Point**")
            fig, ax1 = plt.subplots(figsize=(10, 5))
        
            ax1.set_xlabel("Index")
            ax1.set_ylabel(col1, color="tab:blue")
            ax1.plot(data.index, data[col1], color="tab:blue", label=col1, marker='o')
            ax1.tick_params(axis='y', labelcolor="tab:blue")
        
            ax2 = ax1.twinx()
            ax2.set_ylabel(col2, color="tab:red")
            ax2.plot(data.index, data[col2], color="tab:red", label=col2, marker='x')
            ax2.tick_params(axis='y', labelcolor="tab:red")
        
            # Add vertical line at split
            ax1.axvline(x=split_index, color='black', linestyle='--')
            ax1.text(split_index + 1, ax1.get_ylim()[1]*0.95, 'Split', color='black')
        
            fig.tight_layout()
            st.pyplot(fig)
        
        else:
            st.warning("Please select two different numeric columns.")

        
        # ======================
        # SECTION 5: OLS Regression using Training/Testing Split
        # ======================
        st.write("### 5. OLS Regression (Using Training and Testing Split)")
        
        st.write("#### Select variables for OLS Regression")
        y_col = st.selectbox("**Dependent Variable (Y)**", numeric_columns, key="ols_y")
        
        x_cols = st.multiselect("**Independent Variable(s) (X)**", 
                                [col for col in numeric_columns if col != y_col], 
                                default=[col for col in numeric_columns if col != y_col][:1],
                                key="ols_x")
        
        if y_col and x_cols:
            import statsmodels.api as sm
            from sklearn.metrics import mean_squared_error, r2_score
        
            # Split training and testing data (already defined in section 4)
            X_train = train_data[x_cols]
            X_train = sm.add_constant(X_train)
            y_train = train_data[y_col]
        
            X_test = test_data[x_cols]
            X_test = sm.add_constant(X_test)
            y_test = test_data[y_col]
        
            try:
                model = sm.OLS(y_train, X_train).fit()
                y_pred = model.predict(X_test)
        
                st.write("#### OLS Regression Summary (Training Data)")
                st.text(model.summary())
        
                # Model Evaluation
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
        
                st.write("#### Model Performance on Testing Data")
                st.metric("R-squared (Test Set)", f"{r2:.4f}")
                st.metric("RMSE (Test Set)", f"{rmse:.4f}")
        
                # Optional: Plot Actual vs Predicted
                st.write("**Actual vs Predicted (Test Data)**")
                fig2, ax = plt.subplots()
                ax.plot(y_test.index, y_test, label='Actual', marker='o')
                ax.plot(y_test.index, y_pred, label='Predicted', marker='x')
                ax.legend()
                st.pyplot(fig2)
        
            except Exception as e:
                st.error(f"OLS regression failed: {e}")
        else:
            st.info("Please select a dependent and at least one independent variable for regression.")

        # ======================
    # SECTION 6: Seasonal Decomposition with Independent Train/Test Split
    # ======================
    st.write("### 6. Seasonal Decomposition")
    
    # Column selection
    ts_column = st.selectbox("Select a numeric column for decomposition:", numeric_columns, key="decomp_col")
    
    # Decomposition-specific split
    st.write("**Select a split point for seasonal decomposition (training set)**")
    decomp_split_index = st.slider("Split index for decomposition (before this = training):", 
                                   min_value=1, 
                                   max_value=len(data)-1, 
                                   value=int(len(data)*0.8), 
                                   key="decomp_split")
    
    decomp_train_data = data.iloc[:decomp_split_index]
    decomp_test_data = data.iloc[decomp_split_index:]
    
    # Frequency input
    freq = st.number_input("Enter seasonal frequency (e.g., 12 = yearly seasonality for monthly data):", 
                           min_value=2, 
                           max_value=min(365, len(decomp_train_data)-1), 
                           value=12)
    
    if ts_column:
        try:
            ts_series = decomp_train_data[ts_column]
    
            # Convert index to datetime if possible
            if not pd.api.types.is_datetime64_any_dtype(decomp_train_data.index):
                try:
                    decomp_train_data.index = pd.to_datetime(decomp_train_data.index, errors='coerce')
                except:
                    st.warning("Index could not be converted to datetime. Proceeding with default numeric index.")
    
            ts_series = ts_series.dropna()
    
            decomposition = seasonal_decompose(ts_series, model='additive', period=freq)
    
            st.write(f"**Seasonal Decomposition of `{ts_column}` on Decomposition Training Set ({len(decomp_train_data)} rows)**")
    
            fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
            axes[0].plot(ts_series, label="Original")
            axes[0].set_ylabel("Original")
            axes[0].legend(loc='upper left')
    
            axes[1].plot(decomposition.trend, label="Trend", color="orange")
            axes[1].set_ylabel("Trend")
            axes[1].legend(loc='upper left')
    
            axes[2].plot(decomposition.seasonal, label="Seasonal", color="green")
            axes[2].set_ylabel("Seasonal")
            axes[2].legend(loc='upper left')
    
            axes[3].plot(decomposition.resid, label="Residual", color="red")
            axes[3].set_ylabel("Residual")
            axes[3].legend(loc='upper left')
    
            plt.tight_layout()
            st.pyplot(fig)
    
        except Exception as e:
            st.error(f"Seasonal decomposition failed: {e}")


            
    else:
        st.info("Please upload a file to proceed.")

    st.divider()
