"""
generate_dataset.py

Creates a synthetic customer-support ticket dataset and saves it to
data/tickets.csv. The dataset intentionally contains a few duplicate rows
and a few missing values so that Task 1 (data cleaning) has something
real to clean.

Run:
    python src/generate_dataset.py
"""

import random
from pathlib import Path

import pandas as pd

random.seed(42)

CATEGORIES = {
    "Login Issue": [
        "I am unable to login because my password is not working.",
        "Getting 'invalid credentials' error every time I try to sign in.",
        "My account is locked after multiple failed login attempts.",
        "I forgot my password and the reset email never arrives.",
        "Two factor authentication code is not being accepted.",
        "The login page keeps refreshing without signing me in.",
        "I cannot access my account, it says username not recognized.",
        "OTP for login is not received on my registered mobile number.",
        "App logs me out automatically every few minutes.",
        "Unable to log in using my Google account, keeps failing.",
    ],
    "Application Error": [
        "The application crashes every time I open the reports tab.",
        "I am getting a 500 internal server error on the dashboard.",
        "The app freezes and stops responding when I upload a file.",
        "An error popup appears saying 'unexpected error occurred'.",
        "The page shows a blank white screen after clicking submit.",
        "Application throws a null pointer exception on save.",
        "The mobile app force closes whenever I open notifications.",
        "I keep getting error code 404 when accessing my profile page.",
        "The system displays 'something went wrong' after checkout.",
        "Clicking the export button crashes the entire application.",
    ],
    "Report": [
        "The monthly sales report is not generating any data.",
        "My report shows incorrect totals compared to last month.",
        "Unable to download the report in PDF format.",
        "The chart in the analytics report is not loading properly.",
        "Report export is stuck at 0 percent and never finishes.",
        "The report filters are not applying correctly to the data.",
        "Numbers in the financial report do not match the invoices.",
        "The weekly summary report is missing several entries.",
        "I need help understanding a column in the generated report.",
        "The report scheduler did not send the report on time.",
    ],
    "Account Update": [
        "I want to update my email address on my account.",
        "Please help me change my billing address in the profile.",
        "How do I update my phone number linked to the account?",
        "I need to change my subscription plan to the premium tier.",
        "Requesting a name change on my account due to a legal update.",
        "I want to update my payment method for future invoices.",
        "Unable to edit my company details in the account settings.",
        "Please cancel my current subscription and refund the last charge.",
        "I need to merge two accounts into a single account.",
        "How can I update my profile picture and display name?",
    ],
    "Performance": [
        "The application is taking too long to load every page.",
        "The website is very slow today compared to yesterday.",
        "Search results take more than a minute to appear.",
        "The app lags heavily when scrolling through the list.",
        "Uploading files is extremely slow and often times out.",
        "The dashboard takes forever to refresh after login.",
        "API responses are delayed by several seconds during peak hours.",
        "Video playback keeps buffering even on a fast connection.",
        "The system becomes unresponsive when many users are online.",
        "Page load time has increased a lot since the last update.",
    ],
}

# A handful of harder / ambiguous tickets that borrow vocabulary from more
# than one category. Real support tickets are rarely as clean as the
# templates above, so mixing a few of these in keeps the accuracy realistic
# instead of an unrealistic 100%.
AMBIGUOUS = [
    ("I am unable to generate my report because the app keeps crashing.", "Application Error"),
    ("The report page is loading very slowly and times out.", "Performance"),
    ("After updating my account details the app throws an error.", "Application Error"),
    ("My account dashboard takes too long to load after the update.", "Performance"),
    ("I tried to reset my password but the page crashed midway.", "Application Error"),
    ("Login is extremely slow and sometimes never completes.", "Performance"),
    ("The export button for my report is broken and does nothing.", "Application Error"),
    ("Updating my billing details caused a login error afterwards.", "Login Issue"),
    ("My weekly report keeps freezing halfway through loading.", "Performance"),
    ("I cannot update my profile picture, it just shows an error.", "Application Error"),
    ("The account settings page takes forever to open.", "Performance"),
    ("Report generation failed and now I can't log back in.", "Login Issue"),
]

STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
NAMES = [
    "Aarav Sharma", "Priya Nair", "Rohan Mehta", "Sneha Iyer", "Karan Gupta",
    "Ananya Rao", "Vikram Singh", "Isha Patel", "Arjun Reddy", "Divya Menon",
    "Rahul Verma", "Neha Kapoor", "Sanjay Joshi", "Pooja Desai", "Amitabh Das",
]


PREFIXES = [
    "", "Hi team, ", "Hello, ", "Hi, ", "Dear support, ", "Hi support team, ",
]
SUFFIXES = [
    "", " Please help.", " This is urgent.", " Thanks in advance.",
    " Please assist as soon as possible.", " Kindly look into this.",
    " Let me know how to proceed.",
]


def vary_text(sentence):
    """Add a random prefix/suffix so records in the same category aren't
    all identical strings (avoids unrealistic train/test leakage)."""
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    text = f"{prefix}{sentence}{suffix}"
    return text[0].upper() + text[1:] if text else text


def build_dataset(n_records=180):
    rows = []
    ticket_id = 1000
    categories = list(CATEGORIES.keys())

    while len(rows) < n_records:
        # ~15% of tickets are the harder, ambiguous/overlapping-vocabulary kind
        if random.random() < 0.15:
            base_sentence, category = random.choice(AMBIGUOUS)
        else:
            category = random.choice(categories)
            base_sentence = random.choice(CATEGORIES[category])
        description = vary_text(base_sentence)
        rows.append({
            "ticket_id": f"TCK{ticket_id}",
            "customer_name": random.choice(NAMES),
            "ticket_description": description,
            "date": f"2026-0{random.randint(1,9)}-{random.randint(10,28)}",
            "priority": random.choice(PRIORITIES),
            "status": random.choice(STATUSES),
            "category": category,
        })
        ticket_id += 1

    df = pd.DataFrame(rows)

    # Inject a handful of exact duplicate rows (to exercise duplicate handling)
    dupes = df.sample(6, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)

    # Inject a handful of missing values (to exercise missing-value handling)
    missing_idx = df.sample(8, random_state=2).index
    for i in missing_idx:
        col = random.choice(["ticket_description", "priority", "status"])
        df.loc[i, col] = None

    df = df.sample(frac=1, random_state=3).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = build_dataset()
    Path("data").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/tickets.csv", index=False)
    print(f"Saved {len(df)} records to data/tickets.csv")
    print(df["category"].value_counts())
