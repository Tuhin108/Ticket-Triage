# Customer Support Ticket Classifier

A complete ML-based customer-support ticket classification system that predicts the category of incoming support tickets and provides additional tools for explanation, bulk processing, and response drafting.

## Live Demo

The project is deployed here as it is:

[https://fox15-ticket-triage.hf.space/](https://fox15.pythonanywhere.com/)

The Hugging Face version uses **Gradio**, so its interface is different from the local Flask application.

## What the Project Does

The system classifies customer-support tickets into five categories:

- Login Issue
- Application Error
- Report
- Account Update
- Performance

The project contains two implemented modelling paths:

### Classic Mode
**TF-IDF + Logistic Regression**

This is the primary classification pipeline. It converts ticket descriptions into TF-IDF features and predicts the ticket category using Logistic Regression.

### Smart Mode
**Sentence Embeddings + Logistic Regression**

Smart Mode uses `all-MiniLM-L6-v2` to generate sentence embeddings. In addition to classification, these embeddings are used to retrieve similar historical training tickets using cosine similarity.

This adds an **example-based semantic explanation layer**: a user can see previously classified tickets that are semantically similar to the current ticket.

The project also provides:

- Single-ticket prediction
- Probability estimates
- Similar-ticket explanations in Smart Mode
- Bulk CSV classification
- Flask web interface
- Automatic response suggestions based on predicted category
- Dataset generation and preprocessing
- EDA and model evaluation
- Confusion matrices for both classification approaches

## Model Results

The primary classification pipeline uses TF-IDF features with Logistic Regression
and achieved the following results on the held-out test set of 36 tickets:

| Metric | Score |
|---|---:|
| Accuracy | 94.44% |
| Weighted Precision | 95.15% |
| Weighted Recall | 94.44% |
| Weighted F1 Score | 94.17% |

The project also includes a semantic layer based on the
`all-MiniLM-L6-v2` sentence-transformer.

Rather than being presented as a replacement for the primary classifier, the
semantic layer adds capabilities that TF-IDF alone does not provide:

- Converts ticket descriptions into sentence-level semantic embeddings.
- Retrieves the most semantically similar historical training tickets using
  cosine similarity.
- Provides example-based explanations for Smart Mode predictions.
- Allows users to inspect similar previously classified tickets alongside the
  prediction.
- Supports the same Smart workflow for bulk ticket classification.

The semantic layer was also evaluated on the same test set, but it did not
produce a measurable improvement in classification performance on this
particular dataset. Its main contribution is therefore **semantic similarity
and example-based explanation**, rather than a higher classification score.

The dataset is relatively small and synthetic, so the reported classification
results should not be interpreted as equivalent to production performance on
real-world customer-support data.

## Quick Start

The easiest way to run the complete project is:

```bash
python allinone.py
```

The automation script:

1. Creates or reuses a local virtual environment.
2. Installs `requirements.txt`.
3. Generates `data/tickets.csv` when required.
4. Runs data preprocessing.
5. Trains and evaluates the Classic model.
6. Trains and evaluates the Smart model.
7. Downloads `all-MiniLM-L6-v2` on its first use when required.
8. Runs prediction checks.
9. Starts the Flask application.

Then open:

```text
http://127.0.0.1:5000
```

Press `Ctrl+C` to stop the application.

## Manual Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

### Windows PowerShell

```bash
venv\\Scripts\\Activate.ps1
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Manual Workflow

Run the individual components when needed:

```bash
python src/generate_dataset.py
python src/preprocessing.py
python src/eda.py
python src/train.py
python src/train_embeddings.py
python src/predict.py --test
python src/app.py
```

The Smart model downloads the pretrained sentence-transformer the first time it is used. An internet connection is therefore required for that initial download.

Classic mode can still be used without the pretrained embedding model.

## Application Modes

### Classic Mode

```text
Ticket Description
        ↓
Text Preprocessing
        ↓
TF-IDF
        ↓
Logistic Regression
        ↓
Predicted Category
```

### Smart Mode

```text
Ticket Description
        ↓
Sentence Transformer
(all-MiniLM-L6-v2)
        ↓
Sentence Embedding
        ↓
Logistic Regression
        ↓
Predicted Category

and

Sentence Embedding
        ↓
Cosine Similarity
        ↓
Top Similar Training Tickets
        ↓
Example-Based Explanation
```

The similarity index is built from training tickets, so held-out test tickets are not used as historical explanation examples.

## Bulk Classification

The web application supports batch classification through CSV upload.

The uploaded CSV should contain a `ticket_description` column, or another column whose name contains `description`.

The application processes up to **300 rows per upload**.

## Project Structure

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

Important source files include:

```text
src/
├── generate_dataset.py
├── preprocessing.py
├── eda.py
├── train.py
├── train_embeddings.py
├── embeddings_utils.py
├── explain.py
├── predict.py
└── app.py
```

## Main Technologies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Sentence Transformers
- Flask
- HTML / CSS / JavaScript

## Notes on Reproducibility

The project is designed for local training and experimentation.

The Classic pipeline does not require a pretrained external model.

The Smart pipeline requires the pretrained `all-MiniLM-L6-v2` model on first use. After it has been downloaded and cached, subsequent runs can reuse the local model cache.

The repository contains the code required to reproduce the dataset preparation, EDA, model training, evaluation, prediction, and application workflow.

## Documentation

The detailed project report is available in:

```text
report.pdf
```

It documents the problem, dataset, preprocessing, modelling approach, evaluation, semantic similarity extension, web application, bulk classification, automatic response suggestion, limitations, and future improvements.
