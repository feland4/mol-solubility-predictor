import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator

# -------------------------------------------
# 1. SETUP & MODEL LOADING
# -------------------------------------------
st.set_page_config(page_title="Solubility Predictor", layout="centered")

# Load the trained model
# We use @st.cache_resource so it only loads once (faster)
@st.cache_resource
def load_model():
    with open('solubility_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Initialize the fingerprint generator (Must match your training exactly!)
mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# -------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------
def predict_solubility(smiles):
    # 1. Generate Fingerprint
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, "Invalid SMILES string"
    
    # 2. Convert to Numpy Array (The "Feature")
    fp_array = mfgen.GetFingerprintAsNumPy(mol)
    
    # 3. Reshape for the model (1 sample, 2048 features)
    # The model expects a 2D array, even for one molecule
    X_new = fp_array.reshape(1, -1)
    
    # 4. Predict
    prediction = model.predict(X_new)[0]
    return mol, prediction

# -------------------------------------------
# 3. THE APP INTERFACE
# -------------------------------------------
st.title("🧪 Molecular Solubility Predictor")
st.markdown("""
This app predicts the **Aqueous Solubility (LogS)** of a molecule using an **XGBoost Regressor** trained on the Delaney dataset.
""")

# Input
smiles_input = st.text_input("Enter a SMILES string:", value="CN1C=NC2=C1C(=O)N(C(=O)N2C)C") # Default is Caffeine

if st.button("Predict Solubility"):
    mol, result = predict_solubility(smiles_input)
    
    if mol:
        # Display Prediction
        st.success(f"Predicted LogS: **{result:.3f}**")
        
        # Interpret the result broadly
        if result < -4:
            st.info("Class: Insoluble (Likely needs formulation)")
        elif result < -2:
            st.info("Class: Moderately Soluble")
        else:
            st.info("Class: Soluble")

        # Display Molecule Image
        st.write("### Molecular Structure")
        st.image(Draw.MolToImage(mol), caption=smiles_input)
        
    else:
        st.error("Invalid SMILES string. Please check your input.")

st.markdown("---")
st.markdown("Developed as a Portfolio Project | Powered by RDKit & XGBoost")