import streamlit as st
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import altair as alt

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
        
        # Select only the columns we need to avoid caching issues with unhashable types
        df = df[['molecule_chembl_id', 'canonical_smiles', 'standard_type', 'standard_value']]
        
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
    """
    Generates Morgan fingerprints from a list of SMILES strings.
    Returns a tuple of (fingerprints, valid_indices).
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
    
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints = [morgan_gen.GetFingerprint(mols[i]) for i in valid_indices]
    
    return np.array(fingerprints), valid_indices

def run_pca(fingerprints):
    """Performs PCA on fingerprints to reduce to 2 dimensions."""
    pca = PCA(n_components=2)
    return pca.fit_transform(fingerprints)

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

    st.header("3. Clustering Parameters")
    n_clusters = st.slider("Number of Clusters (KMeans)", 2, 10, 4)
    
    st.header("4. Visualization Thresholds")
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
            
            X, valid_indices = generate_fingerprints(smiles)
            
            if len(X) > 10:
                df_filtered = df_processed.iloc[valid_indices]
                y = df_filtered['p_value'].values
                
                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init to suppress warning
                clusters = kmeans.fit_predict(X)
                
                # Split data while keeping track of molecule info
                X_train, X_test, df_train, df_test = train_test_split(X, df_filtered, test_size=0.2, random_state=42)
                y_train = df_train['p_value'].values
                y_test = df_test['p_value'].values

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
                
                st.header("Bioactivity Landscape Visualization")
                
                st.markdown("""
                The scatter plot below visualizes the chemical space of the test set molecules, reduced to two dimensions using PCA on the Morgan fingerprints. 
                Each point represents a molecule.
                
                - **Color** indicates the cluster each molecule belongs to, based on KMeans clustering of their Morgan fingerprints. This helps in visualizing groups of chemically similar compounds.
                - **Size** corresponds to the prediction error (absolute difference between actual and predicted p-values). Larger points are less accurate predictions, potentially falling outside the SVR model's 'epsilon tube'.
                - **Hover** over a point to see more details about the molecule.
                """)

                # Run PCA on the test set fingerprints
                pca_result = run_pca(X_test)
                
                # Create a DataFrame for plotting
                plot_df = pd.DataFrame({
                    'PC1': pca_result[:, 0],
                    'PC2': pca_result[:, 1],
                    'Actual pValue': y_test,
                    'Predicted pValue': y_pred,
                    'Prediction Error': np.abs(y_test - y_pred),
                    'Molecule ChEMBL ID': df_test['molecule_chembl_id'],
                    'SMILES': df_test['canonical_smiles'],
                    'Cluster': clusters[valid_indices][df_test.index.to_numpy()] # Get cluster for test set molecules
                })
                
                chart = alt.Chart(plot_df).mark_circle().encode(
                    x='PC1',
                    y='PC2',
                    color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category')), # 'N' for nominal data
                    size=alt.Size('Prediction Error', scale=alt.Scale(range=[50, 500])),
                    tooltip=['Molecule ChEMBL ID', 'SMILES', 'Actual pValue', 'Predicted pValue', 'Prediction Error', 'Cluster']
                ).interactive()
                
                st.altair_chart(chart, width='stretch')

            else:
                st.warning("Could not generate valid fingerprints for enough molecules to train a model.")
        
        else:
            st.warning("Not enough data to train a model. Please choose a target with more activities.")
            
    else:
        st.info("No bioactivity data found for the given ChEMBL ID.")
