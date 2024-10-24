import torch
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import DensityMatrix


class Simulator:
    def __init__(self, num_qubits, num_gates):
        self.num_qubits = num_qubits
        self.num_gates = num_gates
        self.gate_to_index = {'h': 0, 's': 1, 'cx': 2}

    def generate_random_clifford_circuit(self):
        qc = QuantumCircuit(self.num_qubits)
        for _ in range(self.num_gates):
            gate = np.random.choice(['h', 's', 'cx'])
            if gate == 'h':
                qubit = np.random.randint(0, self.num_qubits)
                qc.h(qubit)
            elif gate == 's':
                qubit = np.random.randint(0, self.num_qubits)
                qc.s(qubit)
            elif gate == 'cx':
                control = np.random.randint(0, self.num_qubits)
                target = np.random.randint(0, self.num_qubits)
                if control != target:
                    qc.cx(control, target)
        return qc

    def simulate_circuit(self, circuit):
        backend = Aer.get_backend('statevector_simulator')
        job = transpile(circuit, backend)
        job = backend.run(job)
        result = job.result()
        statevector = result.get_statevector()
        return statevector

    def generate_data(self):
        circuit = self.generate_random_clifford_circuit()
        statevector = self.simulate_circuit(circuit)
        return torch.tensor(np.real(statevector)).view(1, -1)

    def generate_guidance_tensor(self, circuit):
        gate_list = []
        for instruction in circuit.data:
            gate_name = instruction[0].name
            qubits = [qubit._index for qubit in instruction[1]]
            gate_index = self.gate_to_index[gate_name]
            gate_list.append((gate_index, qubits))

        num_gates = len(gate_list)
        guidance_tensor = torch.zeros((num_gates, self.num_qubits + 1), dtype=torch.float32)

        for i, (gate_index, qubits) in enumerate(gate_list):
            guidance_tensor[i, 0] = gate_index
            for qubit in qubits:
                guidance_tensor[i, qubit + 1] = 1.0

        return guidance_tensor


def create_totally_mixed_state(d):

    I = [[1 if i == j else 0 for j in range(d)] for i in range(d)]
    I_array = np.array(I)
    rho_mixed = I_array / d
    return DensityMatrix(rho_mixed)

totally_mixed_state = create_totally_mixed_state(3)
simulator = Simulator(num_qubits=3, num_gates=10)
circuit = simulator.generate_random_clifford_circuit()
data_tensor = simulator.generate_data()
guidance_tensor = simulator.generate_guidance_tensor(circuit)

if __name__ == "__main__":
    print(f"Data tensor shape: {data_tensor.shape}")
    print(f"Guidance tensor shape: {guidance_tensor.shape}")
