# SVR Target Bioactivity Dashboard

This Streamlit application allows you to train a Support Vector Regression (SVR) model on bioactivity data from the ChEMBL database and visualize the results.

## Features

- Pulls bioactivity data for a given ChEMBL Target ID.
- Supports IC50, Ki, and EC50 data.
- Generates Morgan fingerprints for molecules using RDKit.
- Trains an SVR model to predict p(Activity).
- Visualizes model predictions with a "tube plot" against biological activity thresholds.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/svr-target.git
    cd svr-target
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application with the following command:

```bash
streamlit run app.py
```

Then, open your web browser to the URL displayed in your terminal.
