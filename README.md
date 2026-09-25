# Customer Support Ticket Classifier

## The project is also deployed in Huggingface by me; the Ui will differ because I am using Gradio in Huggingface - https://fox15-ticket-triage.hf.space/

This repository contains the complete, reproducible application. It classifies
customer-support tickets into five categories and provides a suggested reply.
The web app includes Classic TF-IDF predictions, an optional Smart
sentence-embedding model, confidence breakdowns, explanations, and bulk CSV
classification.

The Hugging Face deployment is maintained separately in the parent project's
`huggingfacedeploy` folder. This folder is the version intended for GitHub,
local development, training, and experimentation.

## Quick start: do everything automatically

From this folder, run:

```bash
python allinone.py
```

`allinone.py` runs the steps in order:

1. Creates or reuses a local `.venv` and switches to it.
2. Installs `requirements.txt`.
3. Generates `data/tickets.csv` if it does not already exist.
4. Prepares the dataset.
5. Trains and evaluates the Classic model.
6. Trains and evaluates the Smart model and downloads
   `all-MiniLM-L6-v2` on its first run.
7. Runs prediction smoke tests.
8. Starts the web app.

Then open http://127.0.0.1:5000. Press `Ctrl+C` to stop it.

## Manual workflow

Create and activate a virtual environment:

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run individual steps when desired:

```bash
python src/generate_dataset.py
python src/preprocessing.py
python src/eda.py
python src/train.py
python src/train_embeddings.py
python src/predict.py --test
python src/app.py
```

The Smart model requires an internet connection the first time it downloads
the pretrained sentence-transformer. If it is unavailable, use
`python allinone.py --skip-smart`; Classic mode remains fully usable.

## Project structure

```text
githubdeploy/
├── allinone.py
├── data/
├── model/
├── screenshots/
├── src/
├── requirements.txt
├── report.md
└── README.md
```

Bulk uploads should contain a `ticket_description` column, or another column
whose name contains `description`. The web app processes up to 300 rows per
upload.
