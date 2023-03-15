from ray.rllib.models.torch.misc import same_padding

print(same_padding((84, 84), (8, 8), 4))
print(same_padding((21, 21), (4, 4), 2))