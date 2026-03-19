from transformers import AutoModel, AutoConfig, AutoTokenizer

BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"

def main():
    # Load config (fastest way to inspect architecture)
    config = AutoConfig.from_pretrained(BASE_MODEL)

    print("Model type:", config.model_type)

    # Most BERT-like models store layer count here
    if hasattr(config, "num_hidden_layers"):
        print("Number of hidden layers:", config.num_hidden_layers)
    else:
        print("num_hidden_layers not found in config")

    # Load full model to inspect detailed architecture
    model = AutoModel.from_pretrained(BASE_MODEL)

    print("\nFull model architecture:\n")
    print(model)

    # If it's a BERT-style model, this usually works:
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        print("\nDetected encoder layers:", len(model.encoder.layer))

if __name__ == "__main__":
    main()