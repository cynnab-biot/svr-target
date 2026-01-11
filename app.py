import streamlit as st
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="SVR Bioactivity Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Functions ---

@st.cache_data
def get_bioactivities(chembl_id):
    """Fetches bioactivity data from ChEMBL for a given target ID."""
    try:
        st.info("Starting data fetch from ChEMBL...") #TODO: Fix issue 
        activity = new_client.activity
        res = activity.filter(target_chembl_id=chembl_id, standard_type__in=["IC50", "Ki", "EC50"])
        df = pd.DataFrame(res)
        st.info(f"Fetched data: {df.shape}")
        return df
    except Exception as e:
        st.error(f"An error occurred while fetching data from ChEMBL: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocesses the bioactivity data."""
    df = df[df['standard_value'].notna()]
    df = df[df['canonical_smiles'].notna()]
    
    # Calculate pX (pIC50, pKi, etc.)
    df['p_value'] = -np.log10(pd.to_numeric(df['standard_value'], errors='coerce') * 1e-9)
    
    df = df.dropna(subset=['p_value'])
    return df

def generate_fingerprints(smiles_list):
    """Generates Morgan fingerprints from a list of SMILES strings."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints = [morgan_gen.GetFingerprint(m) for m in mols if m is not None]
    return np.array(fingerprints)

def plot_predictions(y_true, y_pred, thresholds):
    """Creates a 'tube plot' of predicted vs. actual values with thresholds."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    # Plot thresholds
    for threshold, color in thresholds.items():
        ax.axhline(y=threshold, color=color, linestyle='--')
        ax.axvline(x=threshold, color=color, linestyle='--')

    ax.set_xlabel("Actual pValue")
    ax.set_ylabel("Predicted pValue")
    ax.set_title("SVR Model Predictions vs. Actual Values")
    ax.grid(True)
    
    return fig

# --- Streamlit App ---

st.title("🔬 SVR Bioactivity Dashboard")

st.markdown("""
This app trains a Support Vector Regression (SVR) model on ChEMBL bioactivity data.
Enter a ChEMBL Target ID to get started.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Target Selection")
    chembl_id = st.text_input("Enter ChEMBL Target ID", "CHEMBL1907589") # AChR as default
    
    st.header("2. SVR Parameters")
    C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0)
    epsilon = st.slider("Epsilon", 0.1, 1.0, 0.2)
    
    st.header("3. Visualization Thresholds")
    st.write("Define pValue thresholds for activity.")
    active_threshold = st.number_input("Active Threshold (e.g., > 7)", value=7.0)
    inactive_threshold = st.number_input("Inactive Threshold (e.g., < 5)", value=5.0)


# --- Main Content ---
if chembl_id:
    df = get_bioactivities(chembl_id)
    
    if df is not None and not df.empty:
        st.header(f"Bioactivity Data for {chembl_id}")
        st.write(f"Found {len(df)} activities.")
        
        df_processed = preprocess_data(df.copy())
        
        st.write(f"After preprocessing, {len(df_processed)} activities remain.")
        
        if len(df_processed) > 10:
            st.dataframe(df_processed[['molecule_chembl_id', 'canonical_smiles', 'standard_type', 'standard_value', 'p_value']].head())
            
            # --- Model Training ---
            smiles = df_processed['canonical_smiles'].tolist()
            y = df_processed['p_value'].values
            
            X = generate_fingerprints(smiles)
            
            if len(X) == len(y):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                svr = SVR(C=C, epsilon=epsilon)
                svr.fit(X_train, y_train)
                
                y_pred = svr.predict(X_test)
                
                # --- Display Results ---
                st.header("Model Performance")
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                col1, col2 = st.columns(2)
                col1.metric("R-squared (R²)", f"{r2:.3f}")
                col2.metric("RMSE", f"{rmse:.3f}")
                
                st.header("Prediction Visualization")
                
                st.markdown("""
                The scatter plot below shows the relationship between the actual p-values and the SVR model's predicted p-values.
                
                - **The y=x line (dashed black line)** represents a perfect prediction. Points closer to this line indicate more accurate predictions.
                - **The colored dashed lines** represent the activity thresholds you defined. These can help in visually assessing the model's ability to distinguish between active and inactive compounds.
                """)
                
                thresholds = {
                    active_threshold: "g",
                    inactive_threshold: "r"
                }
                
                fig = plot_predictions(y_test, y_pred, thresholds)
                st.pyplot(fig)
            else:
                st.warning("Could not generate fingerprints for all molecules. The number of fingerprints does not match the number of activities.")
        
        else:
            st.warning("Not enough data to train a model. Please choose a target with more activities.")
            
    else:
        st.info("No bioactivity data found for the given ChEMBL ID.")
