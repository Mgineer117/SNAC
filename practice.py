import torch

a = torch.tensor(11).to(torch.int32)
b = torch.tensor(11.00).long()

# Cast both to integers before comparing
if a == b:
    print("They are equal.")
else:
    print("They are not equal.")

# Cast both to integers before comparing
if int(a) == int(b):
    print("They are equal.")
else:
    print("They are not equal.")
