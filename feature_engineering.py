import pandas as pd


def create_age_group(data):
    bins = [18, 25, 40, 60, 100]
    labels = ["Young", "Adult", "Middle-aged", "Senior"]
    data["Age_Group"] = pd.cut(
        data["alter"], bins=bins, labels=labels, include_lowest=True
    )
    return data


def credit_per_month(data):
    data["Credit_per_Month"] = data["hoehe"] / data["laufzeit"]
    return data


def longterm_loan(data):
    average_duration = data["laufzeit"].mean()
    data["Longterm_Loan"] = (data["laufzeit"] > average_duration).astype(int)
    return data


def feature_engineering_pipeline(data):
    data = create_age_group(data)
    data = credit_per_month(data)
    data = longterm_loan(data)
    return data
