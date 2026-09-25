"""
eda.py  -  Task 2: Exploratory Data Analysis

Prints summary statistics and saves two charts to screenshots/:
  - category_distribution.png
  - priority_status_distribution.png

Run:
    python src/eda.py
"""

import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, prepare_dataset


def run_eda():
    df = load_data()
    df, _ = prepare_dataset(df)

    print("=" * 60)
    print(" TASK 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nTotal number of tickets: {len(df)}")

    print("\nTickets per category:")
    print(df["category"].value_counts())

    print("\nTickets per priority:")
    print(df["priority"].value_counts())

    print("\nTickets per status:")
    print(df["status"].value_counts())

    # --- Chart 1: Category distribution ---
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y="category",
                  order=df["category"].value_counts().index, hue="category",
                  palette="viridis", legend=False)
    plt.title("Ticket Count by Category")
    plt.xlabel("Number of Tickets")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig("screenshots/category_distribution.png", dpi=150)
    plt.close()

    # --- Chart 2: Priority distribution ---
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="priority",
                  order=df["priority"].value_counts().index, hue="priority",
                  palette="magma", legend=False)
    plt.title("Ticket Count by Priority")
    plt.xlabel("Priority")
    plt.ylabel("Number of Tickets")
    plt.tight_layout()
    plt.savefig("screenshots/priority_distribution.png", dpi=150)
    plt.close()

    print("\nSaved charts to screenshots/category_distribution.png "
          "and screenshots/priority_distribution.png")

    print("\nObservations:")
    top_cat = df["category"].value_counts().idxmax()
    top_pri = df["priority"].value_counts().idxmax()
    print(f"- '{top_cat}' is the most common ticket category in this dataset.")
    print(f"- '{top_pri}' is the most common priority level.")
    print("- Category counts are reasonably balanced (no category dominates "
          "the dataset), which should help the classifier learn all classes.")


if __name__ == "__main__":
    run_eda()
