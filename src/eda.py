import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(data, target_column, output_dir):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=target_column, palette="Set2")
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()


def plot_numerical_distribution(data, numerical_columns, output_dir):
    fig, axes = plt.subplots(1, len(numerical_columns), figsize=(6 * len(numerical_columns), 5))
    if len(numerical_columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, numerical_columns):
        sns.histplot(data[column], kde=True, bins=30, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "numerical_distribution.png"))
    plt.close()


def plot_correlation_heatmap(data, output_dir):
    numeric_data = data.select_dtypes(include=["number"])
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()


def plot_boxplot(data, numerical_columns, output_dir):
    fig, axes = plt.subplots(1, len(numerical_columns), figsize=(6 * len(numerical_columns), 5))
    if len(numerical_columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, numerical_columns):
        sns.boxplot(data=data, x=column, ax=ax, palette="Set3")
        ax.set_title(f"Boxplot of {column}")
        ax.set_xlabel(column)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "boxplots.png"))
    plt.close()


def plot_target_vs_features(data, target_column, numerical_columns, output_dir):
    fig, axes = plt.subplots(1, len(numerical_columns), figsize=(6 * len(numerical_columns), 5))
    if len(numerical_columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, numerical_columns):
        sns.boxplot(data=data, x=target_column, y=column, ax=ax, palette="Set2")
        ax.set_title(f"{column} by {target_column}")
        ax.set_xlabel(target_column)
        ax.set_ylabel(column)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "target_vs_numerical.png"))
    plt.close()


def eda_pipeline(data, target_column, numerical_columns, age_column, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Running EDA...")
    plot_class_distribution(data, target_column, output_dir)
    print("  - Class distribution plotted")
    plot_numerical_distribution(data, numerical_columns, output_dir)
    print("  - Numerical distributions plotted")
    plot_correlation_heatmap(data, output_dir)
    print("  - Correlation heatmap plotted")
    plot_boxplot(data, numerical_columns, output_dir)
    print("  - Boxplots plotted")
    plot_target_vs_features(data, target_column, numerical_columns, output_dir)
    print("  - Target vs features plotted")
    print("EDA completed successfully. Figures saved to", output_dir)
