import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)



# Put device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" in a config module and import it everywhere. No hardcoded .cuda(), no hardcoded .cpu(). That one line is what makes the M1 swap and the eventual cloud box no-ops.