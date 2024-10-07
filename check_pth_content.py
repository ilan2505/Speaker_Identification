import torch

model_path = "Convolutional_Speaker_Identification_Log_Softmax_Model4995.pth"
loaded = torch.load(model_path)

# Vérifier si le contenu chargé est un dictionnaire (state_dict)
if isinstance(loaded, dict):
    print("Le fichier semble contenir un state_dict.")
    # Optionnel : afficher les clés pour confirmer
    print("Clés dans le state_dict:", list(loaded.keys()))
else:
    print("Le fichier pourrait contenir un modèle complet.")
    # Optionnel : afficher le type pour aider à identifier
    print("Type de l'objet chargé:", type(loaded))
