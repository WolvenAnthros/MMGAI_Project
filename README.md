## Quantum state generation using circuit-guided diffusion model
Quantum computations and their simulations rely heavily on external strict computational simulators (e.g. **Qiskit**). However, the computational cost of simulations usually makes it hard to evaluate quantum algorithms on a full scale, so that the implementation of a lot of important quantum algorithms remains resctricted. To bypass this issue, in this project the idea of approximating the quantum dynamics with a diffusion model is presented.
***
### Data description
The state of a single qubit (*quantum bit*) can be represented as:
$$|\phi\rangle=\alpha|0\rangle+\beta|1\rangle$$
where
$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

For a multi-qubit system, the general representation of a state is called ***density matrix***. To provide an example, for two qubits it is written as follows:
$
|\psi\rangle\langle\phi| = \begin{pmatrix} \alpha \\ \beta \end{pmatrix} \begin{pmatrix} \gamma^* & \delta^* \end{pmatrix} = \begin{pmatrix} \alpha\gamma^* & \alpha\delta^* \\ \beta\gamma^* & \beta\delta^* \end{pmatrix}
$

The core idea of the diffusion process is to transform the state density matrix into the *totally mixed state* (pure noise in term of quantum state):
$
\rho_t = (1-\bar{\alpha_t})\mathbb{I}/d +\bar{\alpha_t}\rho_0
$,
where $\mathbb{I}/2^N=\sum_i\frac{1}{2^N}|i\rangle\langlei|$.

To help the model learn the underlying dynamics, the fidelity metric between a target and a predicted state is added to the loss function:
$
F(\rho,\sigma) = \tr(\sqrt{\sqrt{\rho}\sigma\sqrt{rho}})^2
$
