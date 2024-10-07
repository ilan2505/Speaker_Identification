import torch
import torchaudio
import os
import pickle
import random
import sys
import constants as c





def process_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    return emission  


speakers_path = "21_to_40/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(str(device) + "\n")

data = {}
all_wav_path = []
num_sublists = c.NUM_SUBLISTS  # number of pickles

bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)

for id in os.listdir(speakers_path):
    print(f'******** {id} ********')
    for sub_dir in os.listdir(f'{speakers_path}/{id}'):
      for wav_file in os.listdir(f'{speakers_path}/{id}/{sub_dir}'):
        file_path = f'{speakers_path}/{id}/{sub_dir}/{wav_file}'
        all_wav_path.append(file_path)
      #for wav_file in os.listdir(f'{speakers_path}/{id}'):
        #file_path = f'{speakers_path}/{id}/{wav_file}'
        #all_wav_path.append(file_path)

all_wav_path_len = len(all_wav_path)

random.shuffle(all_wav_path)

sublist_size = len(all_wav_path) // num_sublists
remainder = len(all_wav_path) % num_sublists

# ===================================== #
# Create sublists and save in txt files #
# ===================================== #
sublist_index = -1
for i in range(0, len(all_wav_path) - remainder, sublist_size):
    sublist = all_wav_path[i:i + sublist_size]
    sublist_index += 1
    if i == 0 and remainder != 0: # added from original and remainder != 0
        # Append remaining elements to the first sublist
        sublist.extend(all_wav_path[-remainder:])

    # save the sublist in txt file
    with open(f'sublist{sublist_index + 1}.txt', 'wb') as f:
        pickle.dump(sublist, f)

all_wav_path = []  # clear from RAM memory

# ===================================== #
# Create from sublists the pickle files #
# ===================================== #
data_counter = [0] * num_sublists

for sublist_index in range(0, num_sublists):

    print(f"Create 'data{sublist_index + 1}.pickle' ...")   #change 1 to 2

    # load the sublist from txt file
    with open(f'sublist{sublist_index + 1}.txt', 'rb') as f:
        sublist = pickle.load(f)
        data_counter[sublist_index] = len(sublist)

    # create tensors
    for file_path in sublist:
        with torch.no_grad():
            if not file_path.endswith("wav"):
                print(file_path)
            emission = process_file(file_path)
            data[file_path] = emission.cpu().numpy()

    # create pickle
    with open(f'data{sublist_index + 1}.pickle', 'wb') as f:    #change 1 to 2
        pickle.dump(data, f)
        data.clear()
        os.remove(f'sublist{sublist_index + 1}.txt')

# check validation
all_sublists_len = 0
for counter in data_counter:
    all_sublists_len += counter

print(f'Valid check --> {all_sublists_len == all_wav_path_len}')

print("Done.")
