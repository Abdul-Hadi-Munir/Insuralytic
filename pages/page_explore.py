import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
BG     = "#F8FAFC"

def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": "#CBD5E1", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E2E8F0", "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
    })


def show(df):
    _style()
    st.header("Explore Patterns: Interactive Data Analysis")
    st.markdown(
        "Use the filters in the sidebar panel below to select a specific subgroup of "
        "policyholders. All charts will update automatically to reflect your selection. "
        "This interactive tool allows you to investigate how different combinations of "
        "demographic and lifestyle characteristics relate to insurance charges."
    )

    # ── Sidebar-style filter panel ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Filter Panel")
    st.markdown("Adjust the controls below to narrow the dataset.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        age_range = st.slider("Age Range", int(df.age.min()), int(df.age.max()),
                              (int(df.age.min()), int(df.age.max())))
        bmi_range = st.slider("BMI Range", float(df.bmi.min()), float(df.bmi.max()),
                              (float(df.bmi.min()), float(df.bmi.max())), step=0.5)
    with fc2:
        sex_sel     = st.multiselect("Sex",    df.sex.unique().tolist(),    default=df.sex.unique().tolist())
        smoker_sel  = st.multiselect("Smoker", df.smoker.unique().tolist(), default=df.smoker.unique().tolist())
    with fc3:
        region_sel  = st.multiselect("Region",   df.region.unique().tolist(), default=df.region.unique().tolist())
        child_sel   = st.multiselect("Children", sorted(df.children.unique().tolist()),
                                     default=sorted(df.children.unique().tolist()))

    # apply filters
    filt = (
        df.age.between(*age_range) &
        df.bmi.between(*bmi_range) &
        df.sex.isin(sex_sel) &
        df.smoker.isin(smoker_sel) &
        df.region.isin(region_sel) &
        df.children.isin(child_sel)
    )
    dff = df[filt].copy()

    st.markdown("---")
    # status bar
    n_total = len(df)
    n_filt  = len(dff)
    pct     = n_filt / n_total * 100

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Records Selected",  f"{n_filt:,} / {n_total:,}")
    sc2.metric("Percentage of Data", f"{pct:.1f}%")
    if n_filt > 0:
        sc3.metric("Average Charges", f"${dff.charges.mean():,.2f}")
        sc4.metric("Median Charges",  f"${dff.charges.median():,.2f}")
    else:
        sc3.metric("Average Charges", "N/A")
        sc4.metric("Median Charges",  "N/A")

    if n_filt < 10:
        st.warning("Fewer than 10 records match the current filters. Widen your selection for meaningful charts.")
        return

    # ── Chart type selector ──────────────────────────────────────────────────
    st.markdown("### Choose a Visualisation")
    chart_type = st.selectbox("Select chart type", [
        "Charges Distribution (Histogram + KDE)",
        "Age vs Charges (Scatter)",
        "BMI vs Charges (Scatter)",
        "Charges by Smoker Status (Box + Violin)",
        "Charges by Region (Bar)",
        "Charges by Number of Children (Bar)",
        "Charges by Sex (Bar)",
        "Pairplot: All Numerical Variables",
        "Feature vs Charges (Custom)",
    ])

    # ── X-axis selector for custom chart ─────────────────────────────────────
    x_feature = None
    if chart_type == "Feature vs Charges (Custom)":
        x_feature = st.selectbox("Select X-axis feature", ["age", "bmi", "children"])

    colour_by = st.selectbox("Colour points by", ["smoker", "sex", "region"], index=0)

    colour_maps = {
        "smoker": {"yes": RED,    "no": BLUE},
        "sex":    {"male": BLUE,  "female": RED},
        "region": {"northeast": BLUE, "northwest": GREEN, "southeast": ORANGE, "southwest": PURPLE}
    }

    st.markdown("---")

    # ── Render charts ─────────────────────────────────────────────────────────
    if chart_type == "Charges Distribution (Histogram + KDE)":
        st.markdown(
            "**What you are seeing:** The histogram shows how frequently charges fall into each "
            "dollar bucket within your filtered subgroup. The KDE curve is a smoothed estimate "
            "of the underlying probability density. The dashed vertical lines mark the mean and "
            "median — a large gap between them indicates skewness."
        )
        from scipy.stats import gaussian_kde
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        axes[0].hist(dff.charges, bins=30, color=BLUE, edgecolor="white", linewidth=0.4, alpha=0.8)
        axes[0].axvline(dff.charges.mean(),   color=RED,   linestyle="--", linewidth=1.5, label=f"Mean ${dff.charges.mean():,.0f}")
        axes[0].axvline(dff.charges.median(), color=GREEN, linestyle=":",  linewidth=1.5, label=f"Median ${dff.charges.median():,.0f}")
        axes[0].set_xlabel("Annual Charges ($)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Histogram — {n_filt:,} records")
        axes[0].legend(fontsize=8)

        vals = dff.charges.values
        kde = gaussian_kde(vals)
        xk  = np.linspace(vals.min(), vals.max(), 300)
        axes[1].fill_between(xk, kde(xk), color=BLUE, alpha=0.3)
        axes[1].plot(xk, kde(xk), color=BLUE, linewidth=2)
        axes[1].axvline(dff.charges.mean(),   color=RED,   linestyle="--", linewidth=1.5)
        axes[1].axvline(dff.charges.median(), color=GREEN, linestyle=":",  linewidth=1.5)
        axes[1].set_xlabel("Annual Charges ($)")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Kernel Density Estimate")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.info(
            f"Selected subgroup — Mean: ${dff.charges.mean():,.2f} | "
            f"Std Dev: ${dff.charges.std():,.2f} | "
            f"Skewness: {dff.charges.skew():.3f}"
        )

    elif chart_type in ("Age vs Charges (Scatter)", "BMI vs Charges (Scatter)", "Feature vs Charges (Custom)"):
        col = {"Age vs Charges (Scatter)": "age",
               "BMI vs Charges (Scatter)": "bmi",
               "Feature vs Charges (Custom)": x_feature}.get(chart_type, x_feature)

        st.markdown(
            f"**What you are seeing:** Each dot represents one policyholder in your "
            f"filtered subgroup. The X-axis shows **{col}** and the Y-axis shows annual "
            f"charges. Points are coloured by **{colour_by}**. "
            "The trend line (OLS regression line) shows the overall direction of the relationship."
        )
        cmap = colour_maps[colour_by]
        c_vals = dff[colour_by].map(cmap)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(dff[col], dff.charges, c=c_vals, alpha=0.5, edgecolors="none", s=25)

        # OLS trend line
        m, b = np.polyfit(dff[col], dff.charges, 1)
        x_line = np.linspace(dff[col].min(), dff[col].max(), 200)
        ax.plot(x_line, m*x_line + b, color="black", linewidth=2, linestyle="--",
                label=f"Trend: charges = {m:,.2f}×{col} + {b:,.0f}")

        legend_handles = [mpatches.Patch(color=v, label=k) for k, v in cmap.items()]
        legend_handles.append(plt.Line2D([0],[0], color="black", linestyle="--", label="OLS Trend"))
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Annual Charges ($)")
        ax.set_title(f"{col.capitalize()} vs Charges — {n_filt:,} records (coloured by {colour_by})")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.info(f"OLS slope: for every 1-unit increase in {col}, charges change by **${m:,.2f}** on average.")

    elif chart_type == "Charges by Smoker Status (Box + Violin)":
        st.markdown(
            "**What you are seeing:** The box plot summarises the median, quartiles, and "
            "outliers for each smoker group. The violin plot adds a kernel density estimate "
            "showing the full shape of the distribution — wider sections indicate more "
            "policyholders at that charge level."
        )
        groups = [dff[dff.smoker==s].charges.values for s in ["no","yes"]]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        bp = axes[0].boxplot(groups, patch_artist=True, labels=["Non-Smoker","Smoker"],
                             medianprops=dict(color="black", linewidth=2),
                             flierprops=dict(marker="o", markersize=3, alpha=0.4))
        for patch, color in zip(bp["boxes"], [GREEN, RED]):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        axes[0].set_ylabel("Charges ($)")
        axes[0].set_title("Box Plot by Smoker Status")

        if all(len(g) > 0 for g in groups):
            parts = axes[1].violinplot(groups, positions=[1,2],
                                       showmedians=True, showextrema=True)
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor([GREEN, RED][i]); body.set_alpha(0.6)
            axes[1].set_xticks([1,2]); axes[1].set_xticklabels(["Non-Smoker","Smoker"])
            axes[1].set_ylabel("Charges ($)")
            axes[1].set_title("Violin Plot by Smoker Status")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        for label, grp in zip(["Non-Smoker","Smoker"], groups):
            if len(grp) > 0:
                st.info(f"**{label}** — n={len(grp):,} | Mean: ${np.mean(grp):,.2f} | Median: ${np.median(grp):,.2f}")

    elif chart_type == "Charges by Region (Bar)":
        st.markdown(
            "**What you are seeing:** Average and median charges for each US region within "
            "your filtered subgroup. Error bars on the average represent one standard deviation."
        )
        rg = dff.groupby("region")["charges"].agg(["mean","median","std","count"]).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pal = [BLUE, GREEN, ORANGE, PURPLE][:len(rg)]
        axes[0].bar(rg.region, rg["mean"], color=pal, edgecolor="black", linewidth=0.5,
                    alpha=0.85, yerr=rg["std"], capsize=4)
        axes[0].set_ylabel("Average Charges ($)")
        axes[0].set_title("Mean Charges by Region (± Std Dev)")
        axes[0].tick_params(axis="x", rotation=15)
        axes[1].bar(rg.region, rg["median"], color=pal, edgecolor="black", linewidth=0.5, alpha=0.85)
        axes[1].set_ylabel("Median Charges ($)")
        axes[1].set_title("Median Charges by Region")
        axes[1].tick_params(axis="x", rotation=15)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    elif chart_type == "Charges by Number of Children (Bar)":
        st.markdown(
            "**What you are seeing:** Average charges broken down by the number of dependants "
            "covered by the policy. This helps assess whether family size is associated with "
            "higher insurance costs."
        )
        cg = dff.groupby("children")["charges"].agg(["mean","count"]).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(cg.children.astype(str), cg["mean"], color=PURPLE,
                      edgecolor="black", linewidth=0.5, alpha=0.85)
        for bar, cnt in zip(bars, cg["count"]):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 150,
                    f"${h:,.0f}\n(n={cnt})", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Number of Children")
        ax.set_ylabel("Average Annual Charges ($)")
        ax.set_title("Average Charges by Number of Children")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    elif chart_type == "Charges by Sex (Bar)":
        st.markdown(
            "**What you are seeing:** A comparison of charge distributions between male and "
            "female policyholders in your selected subgroup."
        )
        sg = dff.groupby("sex")["charges"].agg(["mean","median","std","count"]).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        pal = [BLUE, RED][:len(sg)]
        axes[0].bar(sg.sex, sg["mean"], color=pal, edgecolor="black", linewidth=0.5,
                    alpha=0.85, yerr=sg["std"], capsize=5)
        axes[0].set_ylabel("Average Charges ($)")
        axes[0].set_title("Mean Charges by Sex (± Std Dev)")
        axes[1].bar(sg.sex, sg["median"], color=pal, edgecolor="black", linewidth=0.5, alpha=0.85)
        axes[1].set_ylabel("Median Charges ($)")
        axes[1].set_title("Median Charges by Sex")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    elif chart_type == "Pairplot: All Numerical Variables":
        st.markdown(
            "**What you are seeing:** A pairplot (scatter plot matrix) showing every pair of "
            "numerical variables. The diagonal shows the distribution of each individual variable. "
            "Off-diagonal panels show scatter plots of each pair, coloured by smoker status. "
            "This is a powerful overview chart that can reveal non-obvious interactions."
        )
        num_vars = ["age", "bmi", "children", "charges"]
        n = len(num_vars)
        fig, axes = plt.subplots(n, n, figsize=(14, 14))
        colors_s = dff["smoker"].map({"yes": RED, "no": BLUE})
        for i, var_y in enumerate(num_vars):
            for j, var_x in enumerate(num_vars):
                ax = axes[i][j]
                if i == j:
                    ax.hist(dff[var_x], bins=20, color=BLUE, edgecolor="white",
                            linewidth=0.3, alpha=0.7)
                else:
                    ax.scatter(dff[var_x], dff[var_y], c=colors_s,
                               alpha=0.3, edgecolors="none", s=8)
                if i == n-1: ax.set_xlabel(var_x, fontsize=8)
                if j == 0:   ax.set_ylabel(var_y, fontsize=8)
                ax.tick_params(labelsize=7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        legend_handles = [
            mpatches.Patch(color=RED,  label="Smoker"),
            mpatches.Patch(color=BLUE, label="Non-Smoker")
        ]
        fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
        fig.suptitle("Pairplot of Numerical Variables (coloured by smoker status)",
                     fontweight="bold", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Filtered data table ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("View Filtered Data Table"):
        st.markdown(f"Showing {n_filt:,} records matching current filters.")
        st.dataframe(dff.reset_index(drop=True), use_container_width=True)
        csv = dff.to_csv(index=False)
        st.download_button("Download Filtered Data as CSV", data=csv,
                           file_name="filtered_insurance.csv", mime="text/csv")
