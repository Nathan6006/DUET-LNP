import pandas as pd
import matplotlib.pyplot as plt

def generate_reports_and_distributions(
    df,
    csv_out="class_report.csv",
    fig_prefix="distribution"
):
    report_rows = []

    targets = {
        "toxicity": {
            "value_col": "quantified_toxicity",
            "class_col": "toxicity_class"
        },
        "delivery": {
            "value_col": "quantified_delivery",
            "class_col": "delivery_class"
        }
    }

    for target, cols in targets.items():
        value_col = cols["value_col"]
        class_col = cols["class_col"]


        class_counts = df[class_col].value_counts(dropna=False).sort_index()
        for cls, count in class_counts.items():
            report_rows.append({
                "target": target,
                "class": cls,
                "count": int(count)
            })

        # ---------
        # Continuous distribution plot
        # ---------
        values = df[value_col].dropna()

        plt.figure()
        plt.hist(values, bins=40)
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.title(f"{target.capitalize()} Continuous Distribution")
        plt.tight_layout()
        plt.savefig(f"{fig_prefix}_{target}_continuous.png")
        plt.close()

    # Save class report
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(csv_out, index=False)
def main():
    # however you already create all_df
    all_df = pd.read_csv("../data/all_data.csv")

    generate_reports_and_distributions(
        df=all_df,
        csv_out="class_report.csv",
        fig_prefix="class_dist"
    )

if __name__ == "__main__":
    main()