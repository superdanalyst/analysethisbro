# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 19:20:33 2025

@author: manthis
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 16:43:12 2025

@author: manthis
"""
from eda_tab_test import eda_dashboard_tab
import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")
st.title("Stock Picks Testing (WIP)")

tab1, tab2, tab3 = st.tabs(["Plantation Data", "Textile Data",  "EDA Dashboard"])

with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Plantations_Tea_Auction", type=["xlsx"])
    
    if uploaded_file:
        xls = pd.ExcelFile(uploaded_file)
        df_master = xls.parse("master")
        df_master.columns = df_master.columns.str.strip()
    
        # Ensure columns are numeric and dates clean
        df_master["IQ_TOTAL_REV"] = pd.to_numeric(df_master["IQ_TOTAL_REV"], errors="coerce")
        df_master["Quarterly Tea Auction Revenue"] = pd.to_numeric(df_master["Quarterly Tea Auction Revenue"], errors="coerce")
        df_master["IQ_NET_INC"] = pd.to_numeric(df_master.get("IQ_NET_INC", np.nan), errors="coerce")
        df_master["Period"] = df_master["Period"].astype(str)
    
        ### --- SECTION 1: Revenue vs Auction Revenue Plots ---
        st.header("Reported vs Auction Revenue")
    
        groups = list(df_master.groupby("Ticker"))
        for i in range(0, len(groups), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j >= len(groups):
                    break
                name, group = groups[i + j]
                fig, ax1 = plt.subplots(figsize=(10, 5))
    
                ax1.plot(group["Period"], group["IQ_TOTAL_REV"], marker='o', label="Reported Revenue", color="tab:blue")
                ax1.set_xlabel("Period")
                ax1.set_ylabel("Reported Revenue (LKR Mn)", color="tab:blue")
                ax1.tick_params(axis='y', labelcolor="tab:blue")
                ax1.tick_params(axis='x', rotation=45)
    
                ax2 = ax1.twinx()
                ax2.plot(group["Period"], group["Quarterly Tea Auction Revenue"], marker='x', linestyle='--', label="Auction Revenue", color="tab:red")
                ax2.set_ylabel("Auction Revenue (LKR Mn)", color="tab:red")
                ax2.tick_params(axis='y', labelcolor="tab:red")
    
                ax1.set_title(f"{name}: Reported vs Auction Revenue")
                fig.tight_layout()
                cols[j].pyplot(fig)
    
        ### --- SECTION 2: Correlation Table (Reported vs Auction Revenue) ---
        st.subheader("Correlation: Reported vs Auction Revenue")
        correlations = (
            df_master[["Ticker", "IQ_TOTAL_REV", "Quarterly Tea Auction Revenue"]]
            .dropna()
            .groupby("Ticker")
            .apply(lambda x: x["IQ_TOTAL_REV"].corr(x["Quarterly Tea Auction Revenue"]))
            .reset_index(name="Correlation")
        )
        st.dataframe(correlations)
    
        ### --- SECTION 3: Net Income vs Auction Revenue ---
        st.header("Net Income vs Auction Revenue")
    
        grouped_net = list(df_master.groupby("Ticker"))
        for i in range(0, len(grouped_net), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j >= len(grouped_net):
                    break
                name, group = grouped_net[i + j]
                fig, ax1 = plt.subplots(figsize=(10, 5))
    
                ax1.plot(group["Period"], group["IQ_NET_INC"], marker='o', color="tab:green", label="Net Income")
                ax1.set_xlabel("Period")
                ax1.set_ylabel("Net Income (LKR Mn)", color="tab:green")
                ax1.tick_params(axis='y', labelcolor="tab:green")
                ax1.tick_params(axis='x', rotation=45)
    
                ax2 = ax1.twinx()
                ax2.plot(group["Period"], group["Quarterly Tea Auction Revenue"], marker='x', linestyle='--', color="tab:red", label="Auction Revenue")
                ax2.set_ylabel("Auction Revenue (LKR Mn)", color="tab:red")
                ax2.tick_params(axis='y', labelcolor="tab:red")
    
                ax1.set_title(f"{name}: Net Income vs Auction Revenue (Quarterly)")
                fig.tight_layout()
                cols[j].pyplot(fig)
    
        st.subheader("Correlation: Net Income vs Auction Revenue")
        correlations_net = [
            {"Ticker": name, "Correlation": group["IQ_NET_INC"].corr(group["Quarterly Tea Auction Revenue"])}
            for name, group in grouped_net
        ]
        st.dataframe(pd.DataFrame(correlations_net))
    
        ### --- SECTION 4: Revenue Prediction (Regression Fit) ---
        st.header("Revenue Prediction from Auction Revenue")
    
        results = []
        figures = []
    
        for name, group in df_master.groupby("Ticker"):
            data = group.dropna(subset=["Quarterly Tea Auction Revenue", "IQ_TOTAL_REV", "Period"])
            if len(data) < 4:
                continue
    
            X = data[["Quarterly Tea Auction Revenue"]].values
            y = data["IQ_TOTAL_REV"].values
    
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
    
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            results.append({
                "Ticker": name,
                "R2": r2,
                "RMSE": rmse,
                "Intercept": model.intercept_,
                "Slope": model.coef_[0]
            })
    
            figures.append((name, data["Period"], y, y_pred))
    
        for i in range(0, len(figures), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j >= len(figures):
                    break
                name, dates, y_actual, y_pred = figures[i + j]
                slope = results[i + j]["Slope"]
            
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(dates, y_actual, label="Actual Revenue", marker='o')
                ax.plot(dates, y_pred, label="Predicted Revenue", marker='x')
            
                # Format slope in scientific notation
                slope_str = f"{slope:.2e}"
                ax.set_title(f"{name}: Revenue Over Time\nSlope: {slope_str}", fontsize=12)
            
                ax.set_xlabel("Period")
                ax.set_ylabel("Revenue")
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
                fig.tight_layout()
                cols[j].pyplot(fig)
    
    
        st.subheader("Regression Model Summary")
        st.dataframe(pd.DataFrame(results).sort_values("R2", ascending=False))
        
        ### --- SECTION 5: Net Income Prediction (Regression Fit) ---
        st.header("Net Income Prediction from Auction Revenue")
    
        results = []
        figures = []
    
        for name, group in df_master.groupby("Ticker"):
            data = group.dropna(subset=["Quarterly Tea Auction Revenue", "IQ_NET_INC", "Period"])
            if len(data) < 4:
                continue
    
            X = data[["Quarterly Tea Auction Revenue"]].values
            y = data["IQ_NET_INC"].values
    
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
    
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            results.append({
                "Ticker": name,
                "R2": r2,
                "RMSE": rmse,
                "Intercept": model.intercept_,
                "Slope": model.coef_[0]
            })
    
            figures.append((name, data["Period"], y, y_pred))
    
        for i in range(0, len(figures), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j >= len(figures):
                    break
                name, dates, y_actual, y_pred = figures[i + j]
                slope = results[i + j]["Slope"]
            
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(dates, y_actual, label="Actual Net Income", marker='o')
                ax.plot(dates, y_pred, label="Predicted Net Income", marker='x')
            
                # Format slope in scientific notation
                slope_str = f"{slope:.2e}"
                ax.set_title(f"{name}: Net Income Over Time\nSlope: {slope_str}", fontsize=12)
            
                ax.set_xlabel("Period")
                ax.set_ylabel("Net Income")
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
                fig.tight_layout()
                cols[j].pyplot(fig)
    
    
        st.subheader("Regression Model Summary")
        st.dataframe(pd.DataFrame(results).sort_values("R2", ascending=False))
        
with tab2:
    st.header("Textile Sector Analysis")

    textile_file = st.file_uploader("Upload Textile Sector Data", type=["xlsx"], key="textile")
    if textile_file:
        textile_xls = pd.ExcelFile(textile_file)
        master_df = textile_xls.parse("master")
        master_df.columns = master_df.columns.str.strip()


        # ========== TJL Twin Axes Plots ==========
        plot_df = master_df[["Period", "TJL_TOTAL_REV", "Textiles Exports", "Garments Exports", "Textiles and textile articles Imports"]].dropna()
        plot_df["Lagged Imports (1Q)"] = plot_df["Textiles and textile articles Imports"].shift(1)
        plot_df = plot_df.dropna()

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        fig.suptitle("TJL Revenue vs. Exports and Lagged Imports", fontsize=16)

        axes[0].plot(plot_df["Period"], plot_df["TJL_TOTAL_REV"], marker='o', color='tab:blue')
        axes[0].set_title("Textiles Exports vs. TJL Revenue")
        axes[0].set_ylabel("TJL Revenue (LKR Mn)", color='tab:blue')
        axes[0].tick_params(axis='y', labelcolor='tab:blue')
        ax1b = axes[0].twinx()
        ax1b.plot(plot_df["Period"], plot_df["Textiles Exports"], marker='x', color='tab:green')
        ax1b.set_ylabel("Textiles Exports", color='tab:green')

        axes[1].plot(plot_df["Period"], plot_df["TJL_TOTAL_REV"], marker='o', color='tab:blue')
        axes[1].set_title("Garments Exports vs. TJL Revenue")
        axes[1].set_ylabel("TJL Revenue", color='tab:blue')
        axes[1].tick_params(axis='y', labelcolor='tab:blue')
        ax2b = axes[1].twinx()
        ax2b.plot(plot_df["Period"], plot_df["Garments Exports"], marker='s', color='tab:orange')
        ax2b.set_ylabel("Garments Exports", color='tab:orange')

        axes[2].plot(plot_df["Period"], plot_df["TJL_TOTAL_REV"], marker='o', color='tab:blue')
        axes[2].set_title("Textile Imports vs. TJL Revenue")
        axes[2].set_ylabel("TJL Revenue", color='tab:blue')
        axes[2].tick_params(axis='y', labelcolor='tab:blue')
        axes[2].tick_params(axis='x', rotation=90)
        ax3b = axes[2].twinx()
        ax3b.plot(plot_df["Period"], plot_df["Textiles and textile articles Imports"], marker='^', color='tab:red')
        ax3b.set_ylabel("Imports", color='tab:red')

        st.pyplot(fig)

        # ========== TJL Regression ==========
        filtered_df = plot_df.copy()
        X_train = sm.add_constant(filtered_df[["Textiles Exports", "Garments Exports", "Textiles and textile articles Imports"]])
        y_train = filtered_df["TJL_TOTAL_REV"]
        model = sm.OLS(y_train, X_train).fit()
        filtered_df["Predicted Revenue"] = model.predict(X_train)

        fig2, ax = plt.subplots(figsize=(12, 6))
        ax.plot(filtered_df["Period"], filtered_df["TJL_TOTAL_REV"], label="Actual", marker='o')
        ax.plot(filtered_df["Period"], filtered_df["Predicted Revenue"], label="Predicted", marker='x')
        ax.set_title("TJL Revenue: Actual vs Predicted")
        ax.set_xticklabels(filtered_df["Period"], rotation=90)
        ax.legend()
        st.pyplot(fig2)

        st.subheader("TJL Model Correlations")
        st.write({
            "Textiles Exports": y_train.corr(filtered_df["Textiles Exports"]),
            "Garments Exports": y_train.corr(filtered_df["Garments Exports"]),
            "Imports": y_train.corr(filtered_df["Textiles and textile articles Imports"]),
        })

        # ========== MGT Model ==========
        mgt_df = master_df[["Period", "MGT_IQ_TOTAL_REV", "Textiles Exports", "Garments Exports", "Textiles and textile articles Imports"]].dropna()
        mgt_df["Lagged Imports (1Q)"] = mgt_df["Textiles and textile articles Imports"].shift(1)
        mgt_df = mgt_df.dropna()

        X_mgt = sm.add_constant(mgt_df[["Textiles Exports", "Garments Exports", "Lagged Imports (1Q)"]])
        y_mgt = mgt_df["MGT_IQ_TOTAL_REV"]
        model_mgt = sm.OLS(y_mgt, X_mgt).fit()
        mgt_df["Predicted Revenue"] = model_mgt.predict(X_mgt)

        fig3, ax = plt.subplots(figsize=(12, 6))
        ax.plot(mgt_df["Period"], y_mgt, label="Actual", marker='o')
        ax.plot(mgt_df["Period"], mgt_df["Predicted Revenue"], label="Predicted", marker='x')
        ax.set_title(f"MGT Revenue: Actual vs Predicted (R² = {model_mgt.rsquared:.2f})")
        ax.set_xticklabels(mgt_df["Period"], rotation=90)
        ax.legend()
        st.pyplot(fig3)

        st.subheader("MGT Model Correlations")
        st.write({
            "Textiles Exports": y_mgt.corr(mgt_df["Textiles Exports"]),
            "Garments Exports": y_mgt.corr(mgt_df["Garments Exports"]),
            "Imports": y_mgt.corr(mgt_df["Textiles and textile articles Imports"]),
            "Lagged Imports (1Q)": y_mgt.corr(mgt_df["Lagged Imports (1Q)"]),
        })


with tab3:
    eda_dashboard_tab()
