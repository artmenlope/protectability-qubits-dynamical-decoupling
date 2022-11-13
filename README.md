# protectability-qubits-dynamical-decoupling

Repository containing the Python Jupyter Notebooks used in the article titled "Protectability of IBMQ qubits by dynamical decoupling technique".

The fidelity data is stored in .npy format, which can be loaded into the `all_counts_array` variable of the respective notebook using the `numpy.load` function (these load code lines are already written in the notebooks). The sequence time data stored in the `all_wait_times` variable is generated during the execution of the notebook.

IBM Quantum credentials will be required to run the notebooks.

Package versions:
 - qiskit: 0.37.2
 - qiskit-aer: 0.10.4
 - qiskit-ibmq-provider: 0.19.2
 - qiskit-ignis: 0.7.1
 - qiskit.terra: 0.21.2
 - numpy: 1.23.2
 - matplotlib: 3.5.3
 - scipy: 1.9.0
 - lmfit: 1.0.3

Program (IDE) versions: 
 - Jupyter Notebooks: 6.4.12
 - Spyder: 5.2.2
