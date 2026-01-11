import streamlit as st
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem, Descriptors3D
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
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
        st.info(f"Fetching bioactivity data from ChEMBL for target ID: {chembl_id}") 
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
    df['pIC50'] = -np.log10(pd.to_numeric(df['standard_value'], errors='coerce') * 1e-9)
    
    df = df.dropna(subset=['pIC50'])
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

def calculate_flexibility(smiles):
    """
    Calculates a 3D flexibility score for a molecule.
    Generates a single conformer and computes the Normalized Principal Moments Ratio (NPR).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=AllChem.ETKDG())
    
    if len(cids) == 0:
        return None

    cid = cids[0]
    pmi1 = Descriptors3D.PMI1(mol, confId=cid)
    pmi2 = Descriptors3D.PMI2(mol, confId=cid)
    pmi3 = Descriptors3D.PMI3(mol, confId=cid)
    
    if pmi3 == 0:
        return 0

    npr1 = pmi1 / pmi3
    return npr1


def run_pca(fingerprints):
    """Performs PCA on fingerprints to reduce to 2 dimensions."""
    pca = PCA(n_components=2)
    return pca.fit_transform(fingerprints)

def plot_svr_performance(plot_df, epsilon, high_potency_threshold, low_potency_threshold):
    """Creates an interactive SVR performance plot with tube and potency regions."""
    
    # Add a column to indicate if a point is inside the tube
    plot_df['In Tube'] = np.abs(plot_df['Actual pIC50'] - plot_df['Predicted pIC50']) < epsilon

    # Create the background potency regions
    potency_regions = pd.DataFrame([
        {"x": -1, "x2": low_potency_threshold, "y": -1, "y2": 15, "Potency": "Low"},
        {"x": low_potency_threshold, "x2": high_potency_threshold, "y": -1, "y2": 15, "Potency": "Medium"},
        {"x": high_potency_threshold, "x2": 15, "y": -1, "y2": 15, "Potency": "High"},
    ])
    
    background = alt.Chart(potency_regions).mark_rect().encode(
        x='x:Q',
        x2='x2:Q',
        y='y:Q',
        y2='y2:Q',
        color=alt.Color('Potency:N', scale=alt.Scale(
            domain=['Low', 'Medium', 'High'],
            range=['#FADBD8', '#FDEBD0', '#D5F5E3'] # Light red, yellow, green
        ), legend=alt.Legend(title="Potency Region"))
    )

    # Create the y=x line and SVR tube
    min_val = plot_df[['Actual pIC50', 'Predicted pIC50']].min().min()
    max_val = plot_df[['Actual pIC50', 'Predicted pIC50']].max().max()
    line_data = pd.DataFrame({'x': [min_val, max_val]})
    
    y_x_line = alt.Chart(line_data).mark_line(color='black', strokeDash=[3,3]).encode(x='x', y='x')
    
    tube_upper = alt.Chart(line_data).mark_line(color='gray').encode(
        x='x',
        y=alt.datum.x + epsilon
    )
    tube_lower = alt.Chart(line_data).mark_line(color='gray').encode(
        x='x',
        y=alt.datum.x - epsilon
    )

    # Create the scatter plot of points
    scatter = alt.Chart(plot_df).mark_circle(size=60).encode(
        x=alt.X('Actual pIC50', scale=alt.Scale(domain=[min_val, max_val])),
        y=alt.Y('Predicted pIC50', scale=alt.Scale(domain=[min_val, max_val])),
        color=alt.condition(
            alt.datum.Flexibility > 0,
            alt.Color('Flexibility:Q', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title="Flexibility")),
            alt.value('lightgray')
        ),
        tooltip=['Molecule ChEMBL ID', 'SMILES', 'Actual pIC50', 'Predicted pIC50', 'Prediction Error', 'Flexibility']
    ).interactive()

    return background + y_x_line + tube_upper + tube_lower + scatter

# --- Streamlit App ---

st.title("✨ Potency Estimation with SVR: Structure-Activity Landscape")
st.markdown("""
This application evaluates how well a Support Vector Regression (SVR) model can estimate and rank the potency of compounds based solely on their chemical structure (Morgan fingerprints).

**Understanding pIC50 (Potency Measure):**
- **pIC50** is a common measure of drug potency. It is the negative logarithm of the IC50 value (IC50 is the concentration of an inhibitor where the response is reduced by half).
- **Higher pIC50 values** indicate greater potency (a lower concentration is needed to achieve half-maximal inhibition).
- **Lower pIC50 values** indicate lower potency.
- **Scale interpretation:**
    - **High Potency:** pIC50 > 7 (e.g., IC50 < 100 nM)
    - **Medium Potency:** pIC50 between 5 and 7 (e.g., IC50 between 100 nM and 10 µM)
    - **Low Potency:** pIC50 < 5 (e.g., IC50 > 10 µM)

Enter a ChEMBL Target ID to get started and explore the structure-activity relationship!
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

        if len(df_processed) > 100:
            st.info("ℹ️ Demo Mode: Using only the first 100 molecules for faster processing.")
            df_processed = df_processed.head(100)
        
        if len(df_processed) > 10:
            
            # --- Model Training ---
            smiles = df_processed['canonical_smiles'].tolist()
            
            X, valid_indices = generate_fingerprints(smiles)
            
            if len(X) > 10:
                df_filtered = df_processed.iloc[valid_indices].copy()
                
                y = df_filtered['pIC50'].values
                
                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init to suppress warning
                df_filtered['Cluster'] = kmeans.fit_predict(X)

                st.dataframe(df_filtered[['molecule_chembl_id', 'canonical_smiles', 'standard_type', 'standard_value', 'pIC50']].head())
                
                # Split data while keeping track of molecule info
                X_train, X_test, df_train, df_test = train_test_split(X, df_filtered, test_size=0.2, random_state=42)
                y_train = df_train['pIC50'].values
                y_test = df_test['pIC50'].values

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
                
                # Create a DataFrame for plotting
                plot_df = pd.DataFrame({
                    'Actual pIC50': y_test,
                    'Predicted pIC50': y_pred,
                    'Prediction Error': np.abs(y_test - y_pred),
                    'Molecule ChEMBL ID': df_test['molecule_chembl_id'],
                    'SMILES': df_test['canonical_smiles'],
                    'Cluster': df_test['Cluster']
                })

                # Sort by predicted potency and get top 100
                plot_df_sorted = plot_df.sort_values(by='Predicted pIC50', ascending=False)
                top_100_df = plot_df_sorted.head(100)

                # Calculate flexibility for top 100
                with st.spinner("Calculating 3D flexibility scores for top 100 predictions..."):
                    top_100_smiles = top_100_df['SMILES'].tolist()
                    flexibility_scores = [calculate_flexibility(smiles) for smiles in top_100_smiles]
                    
                    # Use .loc to safely assign values to the new column
                    plot_df['Flexibility'] = np.nan
                    plot_df.loc[top_100_df.index, 'Flexibility'] = flexibility_scores

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
                
                plot_df['PC1'] = pca_result[:, 0]
                plot_df['PC2'] = pca_result[:, 1]
                
                chart_landscape = alt.Chart(plot_df).mark_circle().encode(
                    x='PC1',
                    y='PC2',
                    color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category')), # 'N' for nominal data
                    size=alt.Size('Prediction Error', scale=alt.Scale(range=[50, 500])),
                    tooltip=['Molecule ChEMBL ID', 'SMILES', 'Actual pIC50', 'Predicted pIC50', 'Prediction Error', 'Cluster', 'Flexibility']
                ).interactive()
                
                st.altair_chart(chart_landscape, width='stretch')

                st.header("SVR Performance Analysis")

                st.markdown("""
                The scatter plot below shows the relationship between the actual and predicted pIC50 values.
                
                - The **black dashed line** is the y=x line, representing a perfect prediction.
                - The **gray lines** represent the SVR 'tube' (defined by the `epsilon` parameter). Points inside this tube have zero prediction error according to the SVR model's loss function.
                - The **background colors** indicate the potency regions (low, medium, high).
                - The **point colors** show whether a prediction was inside (blue) or outside (red) the SVR tube.
                """)
                
                chart_svr = plot_svr_performance(plot_df.copy(), epsilon, active_threshold, inactive_threshold)
                st.altair_chart(chart_svr, width='stretch')

            else:
                st.warning("Could not generate valid fingerprints for enough molecules to train a model.")
        
        else:
            st.warning("Not enough data to train a model. Please choose a target with more activities.")
            
    else:
        st.info("No bioactivity data found for the given ChEMBL ID.")
