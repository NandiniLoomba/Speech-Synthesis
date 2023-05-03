import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from util_functions import generate_spectrograms
import params as hp
import librosa

class PrepareDataset(Dataset):
   
    def __init__(self, csv_file, root_dir):
       
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'
        mel, mag = generate_spectrograms(wav_name)
        
        np.save(wav_name[:-4] + '.pt', mel)
        np.save(wav_name[:-4] + '.mag', mag)

        sample = {'mel':mel, 'mag': mag}

        return sample
    
if __name__ == '__main__':
    dataset = PrepareDataset(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
