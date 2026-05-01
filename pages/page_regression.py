import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def show(df):
    st.header("Regression Modeling and Prediction")

    # --- Prepare Data ---
    df_model = df.copy()
    df_model["sex"] = df_model["sex"].map({"male": 1, "female": 0})
    df_model["smoker"] = df_model["smoker"].map({"yes": 1, "no": 0})
    df_model["region"] = df_model["region"].map({
        "northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3
    })

    X = df_model.drop("charges", axis=1)
    y = df_model["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Model Theory ---
    st.subheader("4.1 Multiple Linear Regression Model")
    st.markdown(r"""
    **Model Equation:**

    $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p$$

    Where:
    - $\hat{y}$ is the predicted insurance charge
    - $\beta_0$ is the intercept (baseline cost when all predictors are zero)
    - $\beta_i$ are the regression coefficients representing the marginal effect of each predictor
    - $x_i$ are the input feature values

    The coefficients are estimated using **Ordinary Least Squares (OLS)**, which minimizes the sum of squared residuals:

    $$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
    """)

    # --- Coefficients ---
    st.subheader("4.2 Estimated Coefficients")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)
    coef_df.loc[len(coef_df)] = {"Feature": "Intercept", "Coefficient": model.intercept_}

    st.dataframe(
        coef_df.style.format({"Coefficient": "{:,.2f}"}),
        use_container_width=True
    )

    st.markdown(
        "**Interpretation:** Each coefficient represents the expected change in insurance charges "
        "for a one-unit increase in that feature, holding all other features constant. "
        f"For example, the smoker coefficient of **{model.coef_[X.columns.get_loc('smoker')]:,.2f}** "
        "indicates that being a smoker increases predicted charges by that amount."
    )

    # Coefficient bar chart
    coef_plot = coef_df[coef_df["Feature"] != "Intercept"].sort_values("Coefficient")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if v < 0 else "#3498db" for v in coef_plot["Coefficient"]]
    ax.barh(coef_plot["Feature"], coef_plot["Coefficient"], color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Regression Coefficients")
    ax.axvline(x=0, color="black", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Evaluation Metrics ---
    st.subheader("4.3 Model Evaluation Metrics")

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.markdown(r"""
    **Evaluation Formulas:**

    | Metric | Formula | Interpretation |
    |--------|---------|----------------|
    | R-squared ($R^2$) | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proportion of variance explained (0 to 1) |
    | MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute prediction error |
    | MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared prediction error |
    | RMSE | $\sqrt{MSE}$ | Root of average squared error (same units as target) |
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared", f"{r2:.4f}")
    col2.metric("MAE", f"${mae:,.2f}")
    col3.metric("MSE", f"${mse:,.0f}")
    col4.metric("RMSE", f"${rmse:,.2f}")

    st.markdown(
        f"**Interpretation:** The model explains **{r2*100:.2f}%** of the variance in insurance charges. "
        f"On average, predictions deviate from actual values by **${mae:,.2f}** (MAE). "
        f"The RMSE of **${rmse:,.2f}** indicates the typical magnitude of prediction error."
    )

    # --- Actual vs Predicted ---
    st.subheader("4.4 Actual vs Predicted Values")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred, alpha=0.5, color="#3498db", edgecolors="black", linewidth=0.3, s=20)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Charges ($)")
    axes[0].set_ylabel("Predicted Charges ($)")
    axes[0].set_title(f"Actual vs Predicted (R2 = {r2:.4f})")
    axes[0].legend()

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color="#e67e22", edgecolors="black", linewidth=0.3, s=20)
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted Charges ($)")
    axes[1].set_ylabel("Residuals ($)")
    axes[1].set_title("Residual Plot")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Prediction Form ---
    st.subheader("4.5 Insurance Cost Prediction")
    st.markdown("Enter the details below to obtain an estimated annual insurance charge.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            sex = st.selectbox("Sex", ["male", "female"])
        with col2:
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        with col3:
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

        submitted = st.form_submit_button("Estimate Insurance Cost")

    if submitted:
        sex_enc = 1 if sex == "male" else 0
        smoker_enc = 1 if smoker == "yes" else 0
        region_enc = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}[region]

        input_array = np.array([[age, sex_enc, bmi, children, smoker_enc, region_enc]])
        prediction = model.predict(input_array)[0]

        st.markdown("---")
        st.markdown("### Estimated Annual Insurance Charge")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Predicted Cost", f"${prediction:,.2f}")

        st.markdown(
            f"Based on the provided inputs, the Multiple Linear Regression model estimates "
            f"an annual insurance charge of **${prediction:,.2f}**."
        )

        st.markdown("**Input Summary:**")
        summary_df = pd.DataFrame({
            "Parameter": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
            "Value": [age, sex, bmi, children, smoker, region]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown(
            "*Note: This estimate is derived from a linear model trained on historical data. "
            "Actual insurance costs may vary based on additional factors not captured in this dataset.*"
        )
