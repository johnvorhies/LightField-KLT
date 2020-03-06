# LightField-KLT
LightField-KLT is a Matlab library to implement and test the performance of the 4-D Karhunen-Loeve Transform (KLT) for light field image compression. This library uses the GPU functionality of Matlab to compute the 4-D KLT on NVidia GPUs. If multiple GPUs are available, simultaneous executions of the 4-D KLT can be performed.

## Functions

### mainGPU()
Main control file for testing the performance of the KLT and SKLT at different compression rates. **cRate** can be edited to test different ranges of compression. **rho_s,rho_t,rho_u,rho_v** are the correlation values to use for each dimension. 

### [KLT, SKLT] = generateKLT(rho,N)
generates the KLT or SKLT matrix of size **N** for a given correlation **rho**.

### [DCT, SDCT] = generateDCT(N)
generates the DCT or SDCT matrix of size **N**.

### [st_uv_KLT] = applyKLT_GPU(klt_s,klt_t,klt_u,klt_v,st_uv_single,N)
Performs the 4-D KLT, compression, and inverse 4-D KLT on a 4-D light field. This function requires an NVidia GPU.

#### Inputs:
* klt_s,klt_t,klt_u,klt_v: The transform matrices to be used for each dimension of the light field.
* st_uv_single: A 4-D grayscale light field in the two-plane parameterization form.
* N: The percentage of compression to be used.

#### Outputs:
* st_uv_KLT: A 4-D grayscale light field in the two-plane parameterization form, reconstructed after **N**% compression.
