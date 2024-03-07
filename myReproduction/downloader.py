from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

print(model)

input('Press Enter to continue...')