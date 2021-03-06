# SIMULATIONS OF QUANTUM CIRCUITS ON GPU
## Aim of the code
This code aims to create a very basic simulator of Quantum Circuits on a GPU, using CUDA. The highlight is the Quantum Fourier Transform, and it's usage, or rather it's inverse in the Shor's algorithm. The goal is to show that Quantum Circuits can be simulated well on GPUs, although some possible optimisations may have been missed. Some aspects that could be considered but were not implemented due to lack of time are enumerated later on in this README.

## Usage
The main file is [simulator.cu](./simulator.cu). There is also a notebook named [qiskit_checker](./qiskit_checker.ipynb), which is the file that was used to generate inputs to the simulator, as well as verify the output. The file writes to a certain path, and if it is to be used, please make sure to change the path accordingly. This notebook uses qiskit, and the necessary installs are provided in the beginning of the notebook.

After compiling the code using nvcc (no additional parameters required). The executable must be run with a few arguments from command line. Again please note that not all checks on the arguments are done, so please follow the convention listed below. The following list explains the arguments, based on the first argument which specifies the exact circuit to run:

| Value of first command line argument: | Description and further arguments |
| --------- | ----------- |
|1 | Runs Shor's algorithm to find the prime factors for a number given as second argument. Note that the code is written to be actually execute the circuit, so there is a restriction to the kind of numbers to be provided, being that it must be a product of exactly two primes(both under 100). This is still a very practical case(except the bound); for example RSA encryption works with numbers that are products of 2 primes. ThE algorithm is run in a hybrid manner, more so than the original algorithm. Please contact me in case of issues/doubts regarding this, and please do report bugs.|
|2 | Runs the Quantum Fourier Circuit on a random input state vector and compares with desired output. The second argument must be the input file for the initial state vector, and the third argument is the file with the expected     probabilities of each state. The first line of the input file specifies the number of qubits(n) in the circuit, while the remaining 2^n lines give the probability amplitudes of each state. The file with the expected outputs contains only 2^n lines with the final probabilities(not the amplitudes). |
|3 | Runs an arbitrary Quantum Circuit with a specified initial state. Again, this requires the second argument to be the input file which contains the initial state and the circuit, while the third argument contains the expected probability outputs. The first line of input file contains the number of qubits(n) while the second line contains the number of gates(g). The next 2^n lines contain the probability amplitudes of each state. The following g lines contain the gate to be applied and to which qubit. Each line starts with a number to specify the gate: 1 - Hadamard, 2 - S-Gate, 3 - T-Gate, 4 - C-Not. The reason these were chosen is because they provide one of the simplest "Universal Quantum Gate set": Any Quantum Circuit can be decomposed to containing only these gates such that the output differs by an arbitrary amount in terms of amplitude. For the first 3 gates, the next number (on the same line) provides the qubit the gate should be applied on. For the 4th gate, the Controlled-Not gate, the first parameter specifies the target qubit and the second provides the control qubit. |
|4 | This was mainly for testing the inverse QFT ciruit: it applies the QFT followed by the Inverse QFT, on a random pure state in one of the basis in the 2^4 Hilbert space(only 4 qubits). The final output shows that the same initial state is reached, thus verifying the inverse QFT, of course once the QFT is verified.|

**Example usage: Running the given 10 qubit circuit:**

nvcc simulator.cu

./a.out 3 random_circuit_input_10.txt random_circuit_output_10.txt

**Expected output:**

Applying the random circuit...

Maximum difference in simulated probability and calculated probability = 0.000000

Maximum probability  = 0.007624

Time taken = 0.118553

*(NOTE: The test cases files have the number of qubits at the end of the file name)*

## Some points to consider
The code showed considerable performance improvement in terms of time when compared with Qiskit use on Google COLAB. For example, using Qiskit's State Vector simulator, the time taken to execute and collect the results of a 16 qubit 10000 gate circuit was about 15s, while the CUDA simulator took only 0.1s to complete. Of course the comparison is not so fair because python is interpreted while this code will be compiled. Looking at some papers that used C/C++ simulators instead, show clearly that GPUs are obviously well suited to simulating quantum circuits given the large state space and parallelism. Although care has been taken to provide optimised usage of a GPU, the fact that the code is more generic to running any quantum circuit along with the time constraint, did not allow me to explore every possible avenue. There are still some possible optimisations regarding the usage of shared/global memory which have not been taken care of yet. 

## More info about the code itself
The code is well commented, and is easy enough to understand if one has a basic background in quantum computing. Note that the qubit ordering followed was the same as Qiskit, to make it easier to compare results. The folder also contains an image([4Qubit_QFT.png](./4Qubit_QFT.png)) of the quantum fourier transform circuit for 4 qubits, which was the reference for the QFT code. Also, the maximum number of qubits is currently 16, restricted by the data type used for indexing. This can easily be extended to larger state spaces. If an error like the following occurs(mostly while running on the 16 qubit instances):

*terminate called after throwing an instance of 'thrust::system::system_error'
  what():  parallel_for failed: an illegal memory access was encountered
Aborted (core dumped)*

It is mostly because of the difference in the GPU that was used for developing the code (done on Google Colab which uses a Tesla K80) and the one being used. In such a case run the lower qubit testcases. 

The following 2 papers were used as reference:
1. [Accelerating Shor's Factorization Algorithm on GPUs - I. Savran, M. Demirci, A. H. Yilmaz](https://arxiv.org/abs/1801.01434#:~:text=Shor's%20quantum%20algorithm%20is%20very,much%20faster%20than%20classical%20algorithms.&text=Our%20implementation%20has%2052.5%5Ctimes,20.5%5Ctimes%20speedup%20over%20Liquid.)
2. [https://www.eecg.utoronto.ca/~moshovos/CUDA08/arx/QFT_report.pdf - Alexander Smith Khashayar Khavari](https://www.eecg.utoronto.ca/~moshovos/CUDA08/arx/QFT_report.pdf)

Please do reach out to me (Parth S. Shah, EE17B059, IIT Madras) in case of any issues. 
