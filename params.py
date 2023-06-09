num_mels = 80
n_fft = 2048
sr = 22050
preemphasis = 0.97
frame_shift = 0.0125 
frame_length = 0.05
hop_length = int(sr*frame_shift) 
win_length = int(sr*frame_length) 
n_mels = 80 
power = 1.2 
min_level_db = -100
ref_level_db = 20
hidden_size = 256
embedding_size = 512
max_db = 100
ref_db = 20
n_iter = 60
outputs_per_step = 1
epochs = 100
lr = 0.001
save_step = 2000
image_step = 500
batch_size = 32
cleaners='english_cleaners'
data_path = './data/LJSpeech-1.1'
checkpoint_path = './checkpoint'
sample_path = './generated_audio'