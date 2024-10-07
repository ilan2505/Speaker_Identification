import os
import torch
import torchaudio
import pickle
from cnn_model_wce import Convolutional_Neural_Network  # Replace with your actual model import

# Define the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained wav2vec 2.0 model
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec_model = bundle.get_model().to(device)
wav2vec_model.eval()  # Set the wav2vec model to evaluation mode

def process_file(file_path, model, device):
    waveform, sample_rate = torchaudio.load(file_path)
    with torch.inference_mode():
        outputs = model(waveform.to(device))
        emission = outputs[0]
    return emission.cpu()  # Move the tensor back to CPU

# Load your trained CNN model
model = Convolutional_Neural_Network().to(device)
model_file_path = 'pth_container/Convolutional_Speaker_Identification_Log_Softmax_Model18.pth'
model.load_state_dict(torch.load(model_file_path, map_location=device))
model.eval()

# Path to the directory containing speakers' folders
speakers_dir = 'unknown_folder'

# Initialize a dictionary to track success rates
success_rates = {}

# Iterate over each speaker's directory
for speaker_id in os.listdir(speakers_dir):
    speaker_folder = os.path.join(speakers_dir, speaker_id)
    # Initialize counters for each speaker
    correct_predictions = 0
    total_predictions = 0
    #print(f"speaker_id = {speaker_id}")
    
    for id_folder in os.listdir(speaker_folder):
        id_folder_path = os.path.join(speaker_folder, id_folder)
        if os.path.isdir(id_folder_path):
            for wav_file in os.listdir(id_folder_path):
                wav_file_path = os.path.join(id_folder_path, wav_file)
                #print(f"wav file name : {wav_file_path}")
                if wav_file.endswith('.wav'):
                    emission = process_file(wav_file_path, wav2vec_model, device)
                    emission_tensor = torch.tensor(emission.numpy(), dtype=torch.float).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(emission_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        predicted_label = torch.argmax(output, dim=1)
                        #top_3_indices = torch.topk(probabilities, 3).indices
                        percentages = (torch.nn.functional.softmax(output, dim=1) * 100).tolist()[0]
                        for i, percentage in enumerate(percentages):
                          print(f"Speaker {i+1}: {percentage:.4f}%")
                        if predicted_label.item() + 1 == int(speaker_id):
                            correct_predictions += 1
                        total_predictions += 1
                        print(f'File: {wav_file_path}, Predicted label: {predicted_label.item() + 1}, True Label: {speaker_id}')
    
    # Calculate and store the success rate for the current speaker
    success_rate = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    success_rates[speaker_id] = success_rate
    print(f'Speaker {speaker_id}: {correct_predictions}/{total_predictions} correct predictions, Success Rate: {success_rate:.2f}%')

# Print overall success rates
print("Success Rates per Speaker:")
for speaker_id, rate in success_rates.items():
    print(f'Speaker {speaker_id}: {rate:.2f}% success rate')
