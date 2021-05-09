#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <random>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <numeric>
#include <algorithm>

#define PI 3.14159
#define HADAMARD 1
#define S_GATE 2
#define T_GATE 3
#define CNOT 4
// using namespace std;

/* Parameters needed form external input:
1. Number of qubits (could also define it internally?)
2. 2^N statevector corresponding to the input
*/

// Will use complex values. Thrust provides an inbuilt datatype for this
using comp = thrust::complex<double>;

// Some constants that are used in GPU and CPU. 
__constant__ double sqrt2;
__constant__ int num_qubits;
__constant__ int hs_dim;
int nq, hdim;
int num_blocks, num_threads_per_block;
int N;


// This kernel is to prepare the state for Shor's algorithm. 
__global__ void gpu_prepare_state(comp *gpu_psi, int period, comp init_amp){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    gpu_psi[i] = (i %  (period) == 0 ? (init_amp) : 0.0);
}

// This function is to implement the swap gate between 2 qubits.
// Kernel follows. Always call this kernel with 2**N threads
__device__ void swap_gate(comp *gpu_psi, int i, int a, int b){
    bool ba = (i>>a)&1;
    bool bb = (i>>b)&1;
    if((ba^bb)&&(ba)){
        int j = i^((1<<a)|(1<<b));
        swap(gpu_psi[i], gpu_psi[j]);
    }
}
__global__ void swap_gate_kernel(comp *gpu_psi, int a, int b){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    swap_gate(gpu_psi, i, a, b);
}

// Following functions are to implement hadamard, rotation-Z gates,etc.
// Each function acts on one/two specific states. RZ gates act on only one state
// while the rest act on 2. That's why these are called from kernels with only
// 2**(N-1) threads. This increases the number of qubits we can simulate for 
// with the same number of threads. Also some memory accesses get coalesced,
// And there is no thread divergence
__device__ void hadamard_gate(comp *gpu_psi, int i, int q){
    int i_mq = i^(1<<(q));

    comp temp;
    temp = (1/sqrt2)*(gpu_psi[i_mq] - gpu_psi[i]);
    gpu_psi[i_mq] = (1/sqrt2)*(gpu_psi[i] + gpu_psi[i_mq]);
    gpu_psi[i] = temp;
}

__device__ void rz_gate(comp *gpu_psi, int i, int q, double omega){
    double re = cos(omega);
    double im = sqrt(1 - re*re);
    if(omega < 0){
        im *= -1;
    }
    comp rot = comp(re, im);
    // comp rot = comp(0.0f, omega);
    // rot = exp(rot);
    gpu_psi[i] *= rot;
}

__device__ void x_gate(comp *gpu_psi, int i, int q){
    int ni = i^(1<<(q));
    swap(gpu_psi[i], gpu_psi[ni]);
}

__device__ void cx_gate(comp *gpu_psi, int i, int target_q, int control_q){
    bool bj = (i>>(control_q))&1;
    if(bj){
        x_gate(gpu_psi, i, target_q);
    }
}

__device__ void crz_gate(comp *gpu_psi, int i, int target_q, int control_q, double omega){
    bool bj = (i>>(control_q))&1;
    if(bj){
        rz_gate(gpu_psi, i, target_q, omega);
    }
}

// This is the kernel to be called to implement any one of the "Universal gate set":
// HADAMARD, S-GATE, T-GATE, CONTROLLED-NOT GATE
// This should be lauched with 2**(N-1) threads as explained earlier.
__global__ void apply_gate_kernel(comp *gpu_psi, int gate, int target_q, int control_q, double omega){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int lsb_mask = (1<<target_q)-1;
    int msb_mask = hs_dim - lsb_mask - 1;
    i = ((i&msb_mask)<<1)|(i&lsb_mask);
    i = i|(1<<target_q);
    if(gate == HADAMARD){
        // Apply hadamard gate
        hadamard_gate(gpu_psi, i, target_q);
    }
    else if(gate == S_GATE){
        // Apply S gate
        rz_gate(gpu_psi, i, target_q, PI/2);
    }
    else if(gate == T_GATE){
        // Apply T gate
        rz_gate(gpu_psi, i, target_q, PI/4);
    }
    else if(gate == CNOT){
        //Apply CNOT gate
        cx_gate(gpu_psi, i, target_q, control_q);
    }
}


// Some helper functions that helped during debugging. needed to be called from either
// device or host, so used the __host__ __device__ part
__host__ __device__  void print_state_vector(comp *psi, int hs_dim){
    for(int i=0;i<hs_dim;i++){
        printf("|%d> = %lf + i%lf\n",i, psi[i].real(), psi[i].imag());
    }
}

__host__ __device__  void print_probab(comp *psi, int hs_dim){
    float tp = 0.0f;
    for(int i=0;i<hs_dim;i++){
        float p = pow(thrust::abs(psi[i]),2);
        printf("Probability of state %d = %lf\n",i,p);
        tp += p;
    }
    printf("Total probability = %f\n",tp);
}

// The following 2 kernels are solely for the Quantum Fourier Transform and the Inverse Quantum Fourier Transform
/*
We will be calling the following kernels for each "segment" of the qft or inverse qft. 
Basically doing all the gates on one qubit in each kernel call.
Each kernel call will first do the hadamard gate. All the probability amplitudes will be updated.
After that we will be applying the controlled-rz gates. this will not make updates to all the 
amplitudes, but only to 2^q - 1 amplitudes, where q is the bit we are working on.

Hence, required parameters for each kernel call:
1. Total number of qubits
2. Qubit we are operating on in this kernel call
3. Current statevector (Will need memcpy after every kernel call? instead maybe use unified memory ig?)

*/

__global__ void qft_segment(int cur_qubit, comp *gpu_psi){
    // each thread makes a change to one particular state given by i
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    int lsb_mask = (1<<cur_qubit)-1;
    int msb_mask = hs_dim - lsb_mask - 1;
    i = ((i&msb_mask)<<1)|(i&lsb_mask);
    i = i|(1<<cur_qubit);


    // // First apply the hadamard gate
    // if(i == 0)
    // printf("Applying Hadamard\n");
    
    hadamard_gate(gpu_psi, i, cur_qubit);

    // Then apply the C-ROTs    
    // each state is changed by multiple crz gates so need to loop. 
    // can maybe use dynamic parallelism here
    // think of j representing the control qubit
    
    // if(i == 0)
    //     printf("Applying Phase\n");

    for(int j = 0;j<cur_qubit;j++){
        int k = cur_qubit-j;
        double omega = PI/(1 << k);
        crz_gate(gpu_psi, i, cur_qubit, j, omega);
    }
}

__global__ void inverse_qft_segment(int cur_qubit, comp *gpu_psi){
    // each thread makes a change to one particular state given by i
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    int lsb_mask = (1<<cur_qubit)-1;
    int msb_mask = hs_dim - lsb_mask - 1;
    i = ((i&msb_mask)<<1)|(i&lsb_mask);
    i = i|(1<<cur_qubit);

    // First apply the C-ROTs
    // if(i == 0)
    //     printf("Applying Phase\n");

    for(int j = 0;j<cur_qubit;j++){
        int k = cur_qubit-j;
        double omega = -PI/(1 << k);
        crz_gate(gpu_psi, i, cur_qubit, j, omega);
    }
    
    // // Then apply the hadamard gate
    // if(i == 0)
    // printf("Applying Hadamard\n");
    
    hadamard_gate(gpu_psi, i, cur_qubit);
}


// This is to find the first non-zero probability state for the Shor's algorithm
int measure_first_non_zero(comp *cpu_psi){
    for(int state=1;state<hdim;state++){
        float p = pow(thrust::abs(cpu_psi[state]), 2);
        if(p>1e-3){
            return state;
        }
    }
    return 0;
}

// This is to find the period classically so that we can prepare the state
int get_period(int a, int N){
    int period=1;
    long long x = a;
    while (x != 1LL) {
      x = (a * x) % N;
      ++period;
    }
    return period;
}

// The following 2 functions call the necessary kernels for the implementation of the QFT and Inverse QFT
void inverse_qft(comp *gpu_psi, comp *cpu_psi){
    for(int i=0;i<nq/2;i++){
        swap_gate_kernel<<<num_blocks, num_threads_per_block>>>(gpu_psi, i,nq-1-i);
    }
    cudaDeviceSynchronize();
    // print_probab(cpu_ptr, hdim);
    for(int cur_qubit=0;cur_qubit<nq;cur_qubit++){
        // printf("Working on qubit %d\n",cur_qubit);
        inverse_qft_segment<<<num_blocks, num_threads_per_block/2>>>(cur_qubit, gpu_psi);
        // cudaMemcpy(cpu_psi, gpu_psi, hdim*sizeof(comp), cudaMemcpyDeviceToHost);
        // print_probab(cpu_ptr, hdim);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_psi, gpu_psi, hdim*sizeof(comp), cudaMemcpyDeviceToHost);
    // print_probab(cpu_psi, hdim);
    cudaError_t err = cudaGetLastError();
    printf("\nerror=%d, %s, %s\n\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
}

void qft(comp *gpu_psi, comp *cpu_psi){
    for(int cur_qubit=nq-1;cur_qubit>=0;cur_qubit--){
        // printf("Working on qubit %d\n",cur_qubit);
        qft_segment<<<num_blocks, num_threads_per_block/2>>>(cur_qubit, gpu_psi);
        // cudaMemcpy(cpu_psi, gpu_psi, hdim*sizeof(comp), cudaMemcpyDeviceToHost);
        // print_probab(cpu_ptr, hdim);
    }
    cudaDeviceSynchronize();
    // cudaMemcpy(cpu_psi, gpu_psi, hdim*sizeof(comp), cudaMemcpyDeviceToHost);
    // printf("After h and crot:\n");
    // print_state_vector(cpu_psi, hdim);
    for(int i=0;i<nq/2;i++){
        swap_gate_kernel<<<num_blocks, num_threads_per_block>>>(gpu_psi, i,nq-1-i);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_psi, gpu_psi, hdim*sizeof(comp), cudaMemcpyDeviceToHost);
    // printf("After swap:\n");
    // print_state_vector(cpu_psi, hdim);
    cudaError_t err = cudaGetLastError();
    printf("\nerror=%d, %s, %s\n\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
}


// This function is just to test the Inverse QFT
void test_qft(comp *gpu_psi, comp *cpu_psi, int init_state){
    cpu_psi[init_state] = comp(1.0f,0.0f);
    cudaMemcpy(gpu_psi, cpu_psi, hdim*sizeof(comp), cudaMemcpyHostToDevice);
    // print_state_vector(cpu_psi, hdim);
    qft(gpu_psi, cpu_psi);
    // print_state_vector(cpu_psi, hdim);
    inverse_qft(gpu_psi, cpu_psi);
    // print_state_vector(cpu_psi, hdim);
}

// This function implements the 2nd Quantum part of Shor's algorithm given the period
int shors_algorithm(int a, int period, comp *gpu_psi, comp *cpu_psi){
    const int total_period = ((1 << nq) - 1) / period + 1;
	const comp init_amp = 1.0 / sqrt(total_period);

    gpu_prepare_state<<<1,hdim>>>(gpu_psi, period, init_amp);
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_psi, gpu_psi, hdim*sizeof(comp), cudaMemcpyDeviceToHost);
    // print_probab(cpu_psi, hdim);

    inverse_qft(gpu_psi, cpu_psi);
    int state = measure_first_non_zero(cpu_psi);
    return state;
}


// Main function. There are a few arguments that must be provided, and not all
// error checks are performed. Please read the README for usage.
int main(int argc, char **argv){
    int a;
    srand(time(0));
    struct timeval t1, t2;
    
    double sqrt2_cpu = sqrt(2);
    cudaMemcpyToSymbol(sqrt2, &sqrt2_cpu, sizeof(double), 0, cudaMemcpyHostToDevice);

    if(argc == 1){
        printf("Please choose a circuit to run.\n");
        return 0;
    }
    else{
        int c = std::stoi(argv[1]);
        if(c == 1){
            if(argc !=3){
                printf("Please enter a number to be factorised using Shor's algorithm* !\n");
                return 0;
            }
            N = std::stoi(argv[2]);
            printf("Trying to find factors for %d.\n",N);
            nq = 10;
            hdim = 1<<nq;

            num_blocks = 1 + ((int)hdim/1024);
            num_threads_per_block = min(hdim, 1024);
            
            cudaMemcpyToSymbol(num_qubits, &nq, sizeof(int), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(hs_dim, &hdim, sizeof(int), 0, cudaMemcpyHostToDevice);

            thrust::host_vector<comp> cpu_psi(hdim);
            thrust::device_vector<comp> gpu_psi(hdim);
            comp *gpu_ptr = thrust::raw_pointer_cast(gpu_psi.data());
            comp *cpu_ptr = thrust::raw_pointer_cast(cpu_psi.data());
        
            // Shor's algorithm
            while(1){
                a = rand()%(N-2) + 2;
                printf("Using a = %d\n",a);
                int _gcd = std::__gcd(a, N);
                if (_gcd != 1) {
                    printf("Found factor by fluke = %d. But we want to see shor's quantum algo in action so we shall continue\n",_gcd);
                    continue;
                }
                int period = get_period(a,N);
                printf("Period (found classically) = %d\n",period);
                if(period%2 == 1){
                    printf("Not using this a as we don't want odd period at the end anyway.\n");
                    continue;
                }
                int state = shors_algorithm(a, period, gpu_ptr, cpu_ptr);
                printf("Using continued fractions on state %d \n", state);
                if(!state){
                    printf("Couldn't find proper state to work with.\n");
                    continue;
                }
                // int found_period = test_period_using_cf(a, state);
                int found_period = (int)hdim/state;
                printf("Found period = %d\n", found_period);
                if (found_period) {
                    if (found_period % 2 == 1) {
                        printf("There seems to have been a rounding error :). Actual period is %d\n", --found_period);
                    }
                    const int check = (long int)pow(a, found_period/2)%N;
                    if (check == 1 || check == N-1) {
                      printf("Unfortunately this one doesn't work(Shor's last step is probabilistic).\n");
                      continue;
                    }
                    // find the 2 primes
                    int x = (int)pow(a, found_period/2);
                    int p1 = std::__gcd(x-1, N), p2 = std::__gcd(x+1, N);
                    if(p1 == 1 || p2 == 1){
                        printf("Unfortunately this one doesn't work(Shor's last step is probabilistic).\n");
                        continue;
                    }
              
                    printf("Shor's algorithm result: %d = %d x %d", N, p1, p2);
              
                    return 0;
                }
                // print_probab(cpu_ptr, hdim);
            }
        }

        // run Quantum Fourier Transform on random input
        else if(c == 2){
            FILE *fp;
            if(argc != 4){
                printf("Please provide input AND output files!\n");
                return 0;
            }
            fp = fopen(argv[2],"r");
            fscanf(fp, "%d", &nq);
            hdim = 1<<nq;

            num_blocks = 1 + ((int)hdim/1024);
            num_threads_per_block = min(hdim, 1024);
            
            cudaMemcpyToSymbol(num_qubits, &nq, sizeof(int), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(hs_dim, &hdim, sizeof(int), 0, cudaMemcpyHostToDevice);

            thrust::host_vector<comp> cpu_psi(hdim);
            thrust::device_vector<comp> gpu_psi(hdim);
            comp *gpu_ptr = thrust::raw_pointer_cast(gpu_psi.data());
            comp *cpu_ptr = thrust::raw_pointer_cast(cpu_psi.data());

            for(int s=0;s<hdim;s++){
                double real,imag;
                fscanf(fp, "%lf", &real);
                fscanf(fp, "%lf", &imag);
                cpu_psi[s] = comp(real,imag);
            }
            fclose(fp);
            cudaMemcpy(gpu_ptr, cpu_ptr, hdim*sizeof(comp), cudaMemcpyHostToDevice);
            // print_state_vector(cpu_ptr, hdim);
            qft(gpu_ptr, cpu_ptr);
            // print_probab(cpu_ptr, hdim);
            fp = fopen(argv[3], "r");
            double max_diff = 0.0f;
            double max_p = 0.0f;
            for(int s=0;s<hdim;s++){
                double prob;
                fscanf(fp, "%lf", &prob);
                double p = pow(thrust::abs(cpu_psi[s]),2);
                max_p = max(max_p, p);
                max_diff = max(max_diff, fabs(prob - p));
            }
            printf("Maximum difference in simulated probability and calculated probability = %lf\n",max_diff);
            printf("Maximum probability  = %lf\n", max_p);
        }

        // Run a random circuit
        else if(c == 3){
            FILE *fp;
            if(argc != 4){
                printf("Please provide input AND output files!\n");
                return 0;
            }
            fp = fopen(argv[2],"r");
            fscanf(fp, "%d", &nq);
            hdim = 1<<nq;
            int num_gates;
            fscanf(fp, "%d", &num_gates);

            num_blocks = 1 + ((int)hdim/1024);
            num_threads_per_block = min(hdim, 1024);
            
            cudaMemcpyToSymbol(num_qubits, &nq, sizeof(int), 0, cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(hs_dim, &hdim, sizeof(int), 0, cudaMemcpyHostToDevice);

            thrust::host_vector<comp> cpu_psi(hdim);
            thrust::device_vector<comp> gpu_psi(hdim);
            comp *gpu_ptr = thrust::raw_pointer_cast(gpu_psi.data());
            comp *cpu_ptr = thrust::raw_pointer_cast(cpu_psi.data());

            for(int s=0;s<hdim;s++){
                double real,imag;
                fscanf(fp, "%lf", &real);
                fscanf(fp, "%lf", &imag);
                cpu_psi[s] = comp(real,imag);
            }
            gettimeofday(&t1, 0);
            cudaMemcpy(gpu_ptr, cpu_ptr, hdim*sizeof(comp), cudaMemcpyHostToDevice);
            // print_state_vector(cpu_ptr, hdim);
            printf("Applying the random circuit...\n");
            for(int g=0;g<num_gates;g++){
                int gate;
                fscanf(fp,"%d", &gate);
                int qubit;
                switch(gate){
                    case 1:
                        fscanf(fp, "%d", &qubit);
                        // printf("Applying hadamard on qubit %d..\n", qubit);
                        apply_gate_kernel<<<num_blocks, num_threads_per_block/2>>>(gpu_ptr, HADAMARD,  qubit, 0 , 0);
                        break;
                    case 2:
                        fscanf(fp, "%d", &qubit);
                        // printf("Applying s on qubit %d..\n", qubit);
                        apply_gate_kernel<<<num_blocks, num_threads_per_block/2>>>(gpu_ptr, S_GATE,  qubit, 0, PI/2);
                        break;
                    case 3:
                        fscanf(fp, "%d", &qubit);
                        // printf("Applying t on qubit %d..\n", qubit);
                        apply_gate_kernel<<<num_blocks, num_threads_per_block/2>>>(gpu_ptr, T_GATE, qubit, 0, PI/4);
                        break;
                    case 4:
                        int c_qubit, qubit;
                        fscanf(fp, "%d", &qubit);
                        fscanf(fp, "%d", &c_qubit);
                        // printf("Applying cx on  control qubit %d with target qubit %d..\n", c_qubit, qubit);
                        apply_gate_kernel<<<num_blocks, num_threads_per_block/2>>>(gpu_ptr, CNOT, qubit, c_qubit, 0);
                        break;
                    default: printf("Input file format is wrong!\n");
                            return 0;
                }
            }
            cudaDeviceSynchronize();
            cudaMemcpy(cpu_ptr, gpu_ptr, hdim*sizeof(comp), cudaMemcpyDeviceToHost);
            gettimeofday(&t2, 0);
            // print_state_vector(cpu_ptr, hdim);
            // print_probab(cpu_ptr, hdim);
            fclose(fp);
            fp = fopen(argv[3], "r");
            double max_diff = 0.0f;
            double max_p = 0.0f;
            for(int s=0;s<hdim;s++){
                double prob;
                fscanf(fp, "%lf", &prob);
                double p = pow(thrust::abs(cpu_psi[s]),2);
                max_p = max(max_p, p);
                max_diff = max(max_diff, fabs(prob-p));
            }
            printf("Maximum difference in simulated probability and calculated probability = %lf\n",max_diff);
            printf("Maximum probability  = %lf\n", max_p);
            double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0; // Time taken by kernel in seconds 
            printf("Time taken = %lf", time);
        }

        // Test Inverse QFT. Mainly for debugging purpose.
        else if(c == 4){
            nq = 4;
            hdim = 1<<nq;
            num_blocks = 1 + ((int)hdim/1024);
            num_threads_per_block = min(hdim, 1024);
            thrust::host_vector<comp> cpu_psi(hdim);
            thrust::device_vector<comp> gpu_psi(hdim);
            comp *gpu_ptr = thrust::raw_pointer_cast(gpu_psi.data());
            comp *cpu_ptr = thrust::raw_pointer_cast(cpu_psi.data());
            int init_state = rand()%hdim;
            printf("Starting with state %d\n", init_state);
            test_qft(gpu_ptr, cpu_ptr, init_state);
            print_state_vector(cpu_ptr, hdim);
        }
    }
    return 0;
}
