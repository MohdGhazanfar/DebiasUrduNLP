import streamlit as st
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
import torch
import numpy as np

# Load tokenizer and model
model_path = "."
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path)

# Load debiased embeddings
debiased_embeddings_array = np.load('debiased_embeddings_expanded.npy')
debiased_embeddings_tensor = torch.tensor(debiased_embeddings_array, dtype=torch.float)

# Validate dimensions
original_embeddings = model.get_input_embeddings()
if original_embeddings.weight.shape == debiased_embeddings_tensor.shape:
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(debiased_embeddings_tensor))

# Save the model and tokenizer
save_path = "."
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Initialize the model with the local tokenizer
model = pipeline("fill-mask", model=model, tokenizer=tokenizer)


# Title and header
st.title("Debiased roberta-urdu-small")
st.header("Urdu Language Prediction Model")

# Description
st.markdown("""
This application demonstrates the capabilities of a debiased masked language model 
for Urdu. Enter a sentence with a masked word using `<mask>` to see predicted completions.
""")

# Sidebar for aesthetics
st.sidebar.header("Model Info")
st.sidebar.info("This model predicts missing words in Urdu sentences.")
st.sidebar.text("Model: roberta-urdu-small")
st.sidebar.text("Tokenizer: UrduTokenizer")

# Input field for the user
user_input = st.text_input("Enter a sentence with a masked word:", "")

# Button to generate predictions
if st.button("Generate Predictions"):
    # Get predictions from the model
    predictions = model(user_input)

    # Display predictions
    st.write("Predictions:")
    for pred in predictions:
        st.write(f"- {pred['sequence']} (confidence: {pred['score']:.2f})")


