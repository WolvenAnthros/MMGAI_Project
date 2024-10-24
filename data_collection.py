from base_imports import *
from circuit_simulator import Simulator
from torch.nn.utils.rnn import pad_sequence

class QuantumDataset(Dataset):
    def __init__(self, num_samples, num_qubits, num_gates):
        self.num_samples = num_samples
        self.simulator = Simulator(num_qubits, num_gates)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        circuit = self.simulator.generate_random_clifford_circuit()
        data_tensor = self.simulator.generate_data()
        guidance_tensor = self.simulator.generate_guidance_tensor(circuit)
        return data_tensor, guidance_tensor




# Initialize the dataset and dataloader
# num_samples = 1000
num_qubits = 3
num_gates = 10

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


def collate_fn(batch):
    data_tensors, guidance_tensors = zip(*batch)
    data_tensors = torch.stack(data_tensors)
    guidance_tensors = pad_sequence(guidance_tensors, batch_first=True, padding_value=0.0)
    return data_tensors, guidance_tensors


# Initialize the simulator
# simulator = Simulator()

num_samples = 100  # Number of samples in the dataset
dataset = QuantumDataset(num_samples, num_qubits, num_gates)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

for data_tensors, guidance_tensors in dataloader:
    print(f"Data tensors shape: {data_tensors.shape}")
    print(f"Guidance tensors shape: {guidance_tensors.shape}")
    break  # Just for demonstration, you can remove this in actual usage