# Load model directly
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import torch, torchaudio
import numpy as np

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model.eval()

# Load audio file
audio_input, sample_rate = torchaudio.load("../../Data/Brennan/audio/DownTheRabbitHoleFinal_SoundFile1.wav") # load audio file
# resample audio
audio_input = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_input)
print("audio type:", type(audio_input))
audio_input = audio_input.squeeze()
model_srate = 16000
# Extract features
input_vals = feature_extractor(audio_input, return_tensors="pt", sampling_rate=model_srate, do_normalize=True).input_values
with torch.no_grad():
    model_output = model(input_vals, output_hidden_states=True)
    hidden_states = model_output.get('hidden_states')
        
print('got audio hidden states')
        
wav2vec_emb_sr = hidden_states[0].shape[-2] / (audio_input.shape[-1]/model_srate) # Get the sample rate of the embeddings
        
hidden_states = torch.stack(hidden_states)
hidden_states = hidden_states[-4:].mean(dim=0).squeeze()

embedding_avg = np.transpose(hidden_states)
print("Embedding shape:", embedding_avg.shape)

inv_norms = 1 / (1e-8 + embedding_avg.norm(dim=(1, 2), p=2))
scores = torch.einsum("bt,ot,o->bo", embedding_avg, embedding_avg, inv_norms)

# plot the embeddings
import matplotlib.pyplot as plt
plt.imshow(scores, interpolation="nearest")
plt.show()