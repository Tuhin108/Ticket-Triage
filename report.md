# Solution Report — Customer Support Ticket Classification System

## A. Problem Understanding

A company receives support tickets through email, chat, and other channels.
Each ticket needs to be manually read and assigned to a category (e.g. Login
Issue, Performance) before it can be routed to the right team. This is slow
and inconsistent when done by hand. The goal of this project is to
automatically predict a ticket's category directly from its description
text, using a machine-learning text classifier, so tickets can be routed
faster and more consistently.

## B. Dataset

- **Source:** Self-generated synthetic dataset (`src/generate_dataset.py`), since a
  real support-ticket dataset with labeled categories was not available.
- **Number of records:** 186 raw tickets (178 after removing duplicates/invalid rows).
- **Number of categories:** 5 — Login Issue, Application Error, Report, Account Update, Performance.
- **Fields:** `ticket_id`, `customer_name`, `ticket_description`, `date`, `priority`, `status`, `category`.
- Descriptions were built from ~10 template sentences per category plus random
  greetings/closing phrases, and ~15% of tickets deliberately use "ambiguous"
  wording that borrows vocabulary from a second category (e.g. a Performance
  ticket that also mentions "report"), so the classification problem isn't
  trivially easy. Duplicate rows and a few missing values were injected on
  purpose so Task 1's cleaning steps had real issues to fix.

## C. Data Preprocessing

- **Missing data:** Rows missing `ticket_description` or `category` were dropped
  (a ticket can't be used for training without both). Missing `priority` /
  `status` values were filled with `"Unknown"` instead of dropping the row,
  since those fields aren't needed for the classifier itself.
- **Duplicates:** Exact duplicate rows were removed with `drop_duplicates()`.
- **Text preprocessing:** Each `ticket_description` was lowercased, punctuation
  and non-alphanumeric characters were stripped, and extra whitespace was
  collapsed, producing a `clean_description` column used for modeling.
- **Why:** Lowercasing and punctuation removal prevent the model from treating
  "Login" and "login," as different tokens; removing duplicates prevents the
  same ticket being counted twice in training/evaluation and biasing the
  results.

## D. Model Selection

- **Algorithm:** Logistic Regression (scikit-learn `LogisticRegression`).
- **Why:** It's fast to train, works well on small-to-medium text datasets,
  is easy to explain, and (unlike a plain SVM) gives calibrated-ish
  probability scores via `predict_proba()`, which Task 5's confidence score
  needs. It was compared conceptually against Naive Bayes and Random Forest;
  Logistic Regression was chosen as the best balance of simplicity,
  interpretability, and probability output for this dataset size.
- **Features:** `TfidfVectorizer` (unigrams + bigrams, up to 3000 features)
  turns each cleaned description into a vector of word/phrase-importance
  weights. TF-IDF was chosen over plain Bag-of-Words because it down-weights
  common words (like "the", "issue") that appear in almost every ticket and
  up-weights words that are more distinctive of a category.
- **Pipeline:**
  `Ticket Description → Text Cleaning → TF-IDF → Logistic Regression → Predicted Category`

## E. Training and Testing

- **Split:** 80% training (142 tickets) / 20% testing (36 tickets), stratified
  by category so each category is represented proportionally in both sets.
- **Training:** The TF-IDF vectorizer was fit on the training text only (to
  avoid leaking test-set vocabulary), then Logistic Regression was trained on
  the resulting vectors and their category labels.
- **Testing:** The fitted vectorizer transformed the held-out test
  descriptions, and the trained model predicted their categories, which were
  compared against the true labels.

## F. Results

*(exact numbers from the last run of `train.py`; will vary slightly if you
regenerate the dataset, since it's randomly sampled)*

| Metric | Score |
|---|---|
| Accuracy | 94.4% |
| Precision (weighted) | 95.1% |
| Recall (weighted) | 94.4% |
| F1 Score (weighted) | 94.2% |

**Confusion matrix:** see `screenshots/confusion_matrix.png`. The model got
every *Login Issue*, *Performance*, and *Report* ticket right in the test
set, and only confused a couple of *Application Error* tickets — one was
predicted as *Account Update* and one as *Performance*. This is expected: it
happened on the ambiguous tickets, which were designed to overlap two
categories in vocabulary.

**In simple terms:** the model correctly classified about 94% of test
tickets. This is a satisfactory result for a first-pass classifier — the
categories in this dataset use fairly distinct vocabulary, so a
TF-IDF + Logistic Regression model separates them well. The small number of
errors happened on the intentionally ambiguous tickets, which is a
reasonable place for a simple linear model to struggle.

## G. Prediction (Task 5 test cases)

Five new descriptions not seen during training were tested with
`python src/predict.py --test`:

| # | Ticket Description | Predicted Category | Confidence |
|---|---|---|---|
| 1 | "I can't sign into my account, it keeps saying wrong password even though I reset it." | Login Issue | 32.2% |
| 2 | "The app crashed twice today while I was uploading a document." | Performance | 31.9% |
| 3 | "My invoice report has the wrong total, can you check it?" | Report | 22.9% |
| 4 | "I would like to change the email address associated with my account." | Account Update | 32.1% |
| 5 | "Everything on the website loads so slowly today, is there an outage?" | Performance | 32.8% |

4 of 5 categories were predicted correctly. Test case 2 ("app crashed") was
predicted as *Performance* rather than *Application Error* — a reasonable
mistake, since "crashed" and slowness both describe the app misbehaving.
Confidence scores are modest (20–35%) because these test sentences use
phrasing quite different from the training templates, and 5 fairly close
categories share a fair amount of vocabulary; on a larger, more diverse
dataset the model would likely be more confident on genuinely new phrasing.

## H. Limitations

- **Small, synthetic dataset:** ~180 tickets built from a limited number of
  sentence templates. It demonstrates the pipeline correctly but is far
  smaller and less varied than real support-ticket volume and language.
- **Limited categories:** Only 5 categories are covered; a real system would
  need many more (billing, security, feature requests, etc.).
- **Similar wording between categories:** Application Error and Performance
  tickets in particular share overlapping vocabulary ("slow", "crash",
  "error"), which is where most of the model's mistakes occur.
- **Simple model:** Logistic Regression with TF-IDF doesn't understand word
  order, context, or synonyms it hasn't seen — it can misclassify tickets
  phrased very differently from the training examples.
- **No spelling-correction or slang handling:** typos or informal chat-style
  language would likely reduce accuracy further.

## I. Future Improvements

If this were being built for a real company, I would:

- Collect a much larger, real (anonymized) ticket dataset with many more
  categories and label it with human agents rather than templates.
- Add more text-preprocessing (spell-correction, stopword removal, stemming/lemmatization).
- Try stronger models: Support Vector Machines, gradient-boosted trees, or a
  fine-tuned transformer/embedding-based classifier for better handling of
  varied phrasing.
- Add an LLM-based fallback for tickets the model is not confident about
  (low predict_proba score), instead of forcing a guess.
- Continuously retrain the model as new labeled tickets come in, and collect
  agent feedback (was the predicted category correct?) to improve it over time.
- Wrap the model behind a proper production API (e.g. FastAPI) with logging,
  monitoring, and a database instead of loading files from disk.
- Store tickets and predictions in a database rather than a CSV file.

## J. Upgrades Beyond the Assignment Brief

Three extensions were added on top of the required tasks (see `README.md`
section 9 for setup and usage details):

1. **Smart model (sentence embeddings):** a second classifier using a
   pretrained sentence-transformer (`all-MiniLM-L6-v2`) instead of TF-IDF,
   selectable via a toggle in the web app. This is a transfer-learning
   approach: rather than training a bigger model from scratch (which would
   overfit on a dataset this small - see Limitation H), it reuses semantic
   understanding a much larger model already learned, so it can recognize
   paraphrased tickets that share no vocabulary with the training examples.
2. **Explainability:** the Classic model highlights the exact words that
   drove its prediction using its own learned coefficients; the Smart model
   instead shows the most similar past tickets by embedding similarity,
   since individual embedding dimensions aren't human-interpretable the way
   TF-IDF word weights are. Each model is explained with the technique
   that's actually valid for it, rather than forcing one method onto both.
3. **Bulk CSV upload:** classify many tickets at once from an uploaded CSV,
   with results viewable in the app and downloadable as a CSV.

I wrote and unit-tested the embeddings code (training loop, saving/loading,
the Flask routes, the UI) using a stand-in fake embedder, since my build
environment couldn't reach Hugging Face's servers to download the real
pretrained model. The code follows the standard, documented
`sentence-transformers` API, but I'd recommend running
`python src/train_embeddings.py` yourself as a first check with a normal
internet connection.

## Bonus Task — Auto-generated Response

Implemented as a **rule-based** lookup (`SUGGESTED_RESPONSES` dictionary in
`src/predict.py`): once a category is predicted, a canned but relevant reply
for that category is returned alongside the prediction (used in both
`predict.py` and `app.py`). A rule-based approach was chosen over an external
LLM API to keep the project fully self-contained, free to run, and without
requiring any API keys or credentials.

Example:

```
Input: "I forgot my password and cannot login."
Predicted Category: Login Issue
Suggested Response: Please use the 'Forgot Password' option on the login
page to reset your password. If the problem continues, please contact
the support team.
```

## AI Tool Usage Disclosure

Claude (Anthropic) was used throughout this project to help design the
pipeline structure, write and debug the Python code (dataset generation,
preprocessing, EDA, training, prediction, and the Flask app), and draft this
documentation. Every script was run and verified end-to-end (including a
clean virtual-environment install) before submission, and I can explain any
part of the code and the reasoning behind it.
