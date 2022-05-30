import torch

pth_path = './pretrained_models/vqvae_bottom.pth'
pretrained_models = torch.load(pth_path)

print(pretrained_models.keys())

state_dict = {}
for key in pretrained_models.keys():
    state_dict[key.replace('mid_', 'bot_')] = pretrained_models[key]

torch.save(state_dict, './pretrained_models/vqvae_bottom_new.pth')
