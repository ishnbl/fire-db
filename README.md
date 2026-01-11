# FireDB

FireDB is a simple GPU-accelerated vector database.
I came with the idea to do this while building my BYOP project, my project involved a KNN and visualising latent spaces I thought that it would be cool to have a simple Vector DB where you can just toss in a numpy file and do KNN, t-SNE, PCA etc on vectors directly, I have added the functionality of KNN and will soon add t-SNE, PCA, UMAP etc
It stores vectors on disk, uploads them to the GPU, and performs exact L2 nearest-neighbor search using CUDA and cuBLAS.The project used memory mapping to reduce the amount of ram required
The DB supports importing vectors from raw numpy files, and gives you the ability to do fast KNN search on it by directly loading it to your GPU Memmory and Doing matrix multiplication.
The DB is very fast as long as you can fit everything in your GPU VRAM, hence its usefull for people who are dealing with around 1-3 Million Vectors (possible to load on a consumer laptop GPU),however it becomes slower as compared to HNSW style algorithms used by FAISS etc.
## Feature

* Exact L2 similarity search
* GPU-accelerated using CUDA + cuBLAS
* Persistent on-disk storage
* Incremental vector insertion
* KNN search on entire batches using matmul
* NumPy vector import
* Simple CLI
* Memory Mapping so that the program can lazy load vectors, even if the size of vectors is more than your system RAM.
## Requirements

**Hardware**
* NVIDIA GPU with CUDA support

**Software**
* Linux 
* CUDA Toolkit (11.x or newer)
* GCC / G++ (C++17)
* cuBLAS (comes with CUDA)

## Project Structure

```text
.
├── main.cpp
├── src/
│   └── core/
│       ├── gpu.h        # GPU index (CUDA + cuBLAS)
│       ├── slab.h       # Vector storage
