#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>

// --- Data Structures ---
// Using CUDA's built-in float3


struct Particle {
    float3 position;
    float3 velocity;
    float mass;
    float radius;
};

struct SimulationParams {
    int numParticles;
    float boxSize;
    float gridSize;
    float gravity;
    float dragCoefficient;
    float pairwiseForceStrength;
    float cutoffDistance;
    float maxTimeStep;
    float minTimeStep;
    float safetyFactor;
    float elasticity;
    int maxIterations;
    int printInterval;
};

// Function Prototypes
void initializeParticles(std::vector<Particle>& particles, const SimulationParams& params);
void runSimulation(std::vector<Particle>& particles, const SimulationParams& params);
void printDeviceInfo();

//--- CUDA Kernels ---

__global__ void calculateForcesKernel(float3* pos, float3* vel, float* mass, float* radius, float3* force, int* grid, int* cellStart,
                                     int numCellsX, int numCellsY, int numCellsZ,
                                     int numParticles, float boxSize, float gridSize, float gravity,
                                     float dragCoefficient, float pairwiseForceStrength, float cutoffDistance) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float3 f_grav = {0.0f, -gravity, 0.0f};
    float3 f_drag = {-dragCoefficient * vel[i].x, -dragCoefficient * vel[i].y, -dragCoefficient * vel[i].z};

   force[i].x = f_grav.x + f_drag.x;
   force[i].y = f_grav.y + f_drag.y;
   force[i].z = f_grav.z + f_drag.z;

    // Get cell position
    int cellX = (int) ((pos[i].x + boxSize / 2.0f) / gridSize);
    int cellY = (int) ((pos[i].y + boxSize / 2.0f) / gridSize);
    int cellZ = (int) ((pos[i].z + boxSize / 2.0f) / gridSize);

    if (cellX < 0 || cellX >= numCellsX || cellY < 0 || cellY >= numCellsY || cellZ < 0 || cellZ >= numCellsZ) return; // Particle out of bounds

    int cellIndex = cellX + cellY * numCellsX + cellZ * numCellsX * numCellsY;
    int start = cellStart[cellIndex];
    int end = (cellIndex == (numCellsX * numCellsY * numCellsZ - 1)) ? numParticles : cellStart[cellIndex+1];

    for (int j = start; j < end; j++)
    {
        if (i == j) continue; // Skip self interaction
        float3 r_ij;
        r_ij.x = pos[j].x - pos[i].x;
        r_ij.y = pos[j].y - pos[i].y;
        r_ij.z = pos[j].z - pos[i].z;

        float dist_sq = r_ij.x*r_ij.x + r_ij.y*r_ij.y + r_ij.z*r_ij.z;

        if (dist_sq < cutoffDistance * cutoffDistance)
        {
          float dist = sqrtf(dist_sq);
          float mag = pairwiseForceStrength/(dist*dist);

          float3 f_ij;
          f_ij.x = mag * r_ij.x / dist;
          f_ij.y = mag * r_ij.y / dist;
          f_ij.z = mag * r_ij.z / dist;

          force[i].x += f_ij.x;
          force[i].y += f_ij.y;
          force[i].z += f_ij.z;
        }
    }
    // Iterate over neighboring cells
    for(int dx = -1; dx<=1; dx++)
    {
        for(int dy = -1; dy<=1; dy++)
        {
            for (int dz = -1; dz<=1; dz++)
            {
                if (dx==0 && dy==0 && dz==0) continue; // Skip self cell
                int neighborX = cellX + dx;
                int neighborY = cellY + dy;
                int neighborZ = cellZ + dz;

                if (neighborX < 0 || neighborX >= numCellsX || neighborY < 0 || neighborY >= numCellsY || neighborZ < 0 || neighborZ >= numCellsZ) continue;

                int neighborIndex = neighborX + neighborY * numCellsX + neighborZ * numCellsX * numCellsY;
                int start = cellStart[neighborIndex];
                int end = (neighborIndex == (numCellsX * numCellsY * numCellsZ - 1)) ? numParticles : cellStart[neighborIndex+1];

                for (int j = start; j < end; j++)
                {
                    if (i == j) continue; // Skip self interaction
                    float3 r_ij;
                    r_ij.x = pos[j].x - pos[i].x;
                    r_ij.y = pos[j].y - pos[i].y;
                    r_ij.z = pos[j].z - pos[i].z;

                    float dist_sq = r_ij.x*r_ij.x + r_ij.y*r_ij.y + r_ij.z*r_ij.z;

                    if (dist_sq < cutoffDistance * cutoffDistance)
                    {
                    float dist = sqrtf(dist_sq);
                    float mag = pairwiseForceStrength/(dist*dist);

                    float3 f_ij;
                    f_ij.x = mag * r_ij.x / dist;
                    f_ij.y = mag * r_ij.y / dist;
                    f_ij.z = mag * r_ij.z / dist;

                    force[i].x += f_ij.x;
                    force[i].y += f_ij.y;
                    force[i].z += f_ij.z;
                    }
                }
           }
       }
   }
}


__global__ void integrateKernel(float3* pos, float3* vel, float3* force, float* mass, float * radius, int numParticles, float dt, float elasticity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

     // Update velocity and position
    float3 acc = {force[i].x / mass[i], force[i].y / mass[i], force[i].z / mass[i]};

    vel[i].x += acc.x * dt;
    vel[i].y += acc.y * dt;
    vel[i].z += acc.z * dt;

    pos[i].x += vel[i].x * dt;
    pos[i].y += vel[i].y * dt;
    pos[i].z += vel[i].z * dt;


    // Collision Detection and Response
    for (int j = 0; j < numParticles; ++j) {
        if (i == j) continue;
        float3 r_ij = {pos[j].x - pos[i].x, pos[j].y - pos[i].y, pos[j].z - pos[i].z};
        float dist_sq = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
        float radiiSum = radius[i] + radius[j];

        if (dist_sq < radiiSum * radiiSum) {
            float dist = sqrtf(dist_sq);

            float3 normal = {r_ij.x / dist, r_ij.y / dist, r_ij.z / dist};

             // Relative velocity
            float3 v_rel = {vel[i].x - vel[j].x, vel[i].y - vel[j].y, vel[i].z - vel[j].z};

            // Compute the dot product of the relative velocity and the collision normal
            float v_rel_dot_normal = v_rel.x * normal.x + v_rel.y * normal.y + v_rel.z * normal.z;

            if(v_rel_dot_normal > 0) continue; // Particles are moving away from each other.

            // Impulse
            float impulse = -(1 + elasticity) * v_rel_dot_normal / (1.0f / mass[i] + 1.0f / mass[j]);

            // Apply the impulse
            vel[i].x += impulse * normal.x / mass[i];
            vel[i].y += impulse * normal.y / mass[i];
            vel[i].z += impulse * normal.z / mass[i];
            vel[j].x -= impulse * normal.x / mass[j];
            vel[j].y -= impulse * normal.y / mass[j];
            vel[j].z -= impulse * normal.z / mass[j];


             //Ensure particles do not remain stuck
            float penetration = radiiSum - dist;
            pos[i].x += 0.5f * normal.x * penetration;
             pos[i].y += 0.5f * normal.y * penetration;
             pos[i].z += 0.5f * normal.z * penetration;
            pos[j].x -= 0.5f * normal.x * penetration;
            pos[j].y -= 0.5f * normal.y * penetration;
            pos[j].z -= 0.5f * normal.z * penetration;
            }
       }

}

//--------------------------------------------------------------------


//--------------------------------------------------------------------

void runSimulation(std::vector<Particle>& particles, const SimulationParams& params) {
    // --- Device Memory Allocation ---
    float3* d_pos[2]; // Double buffering for positions
    float3* d_vel[2]; // Double buffering for velocities
    float* d_mass;
    float* d_radius;
    float3* d_force;
    int* d_grid;
    int* d_cellStart;

    size_t particleMemSize = params.numParticles * sizeof(float3);
    size_t massMemSize = params.numParticles * sizeof(float);

    cudaMalloc((void**)&d_pos[0], particleMemSize);
    cudaMalloc((void**)&d_pos[1], particleMemSize);
    cudaMalloc((void**)&d_vel[0], particleMemSize);
    cudaMalloc((void**)&d_vel[1], particleMemSize);
    cudaMalloc((void**)&d_mass, massMemSize);
    cudaMalloc((void**)&d_radius, massMemSize);
    cudaMalloc((void**)&d_force, particleMemSize);


    //--- Spatial Partitioning
    int numCellsX = (int)(params.boxSize / params.gridSize);
    int numCellsY = (int)(params.boxSize / params.gridSize);
    int numCellsZ = (int)(params.boxSize / params.gridSize);
    int numCells = numCellsX * numCellsY * numCellsZ;

    cudaMalloc((void**)&d_grid, params.numParticles * sizeof(int));
    cudaMalloc((void**)&d_cellStart, (numCells+1) * sizeof(int));


   std::vector<int> grid_host(params.numParticles);
    std::vector<int> cellStart_host(numCells+1,0);

    // --- Transfer particle data to device ---
    for (int i = 0; i < params.numParticles; i++) {
      grid_host[i] = (int) ((particles[i].position.x + params.boxSize/2.0f) / params.gridSize) +
                     (int) ((particles[i].position.y + params.boxSize/2.0f) / params.gridSize) * numCellsX +
                     (int) ((particles[i].position.z + params.boxSize/2.0f) / params.gridSize) * numCellsX * numCellsY;
    }

    // Prefix Sum to calculate start of each cell
    std::vector<int> cellSizes(numCells,0);
    for (int i = 0; i < params.numParticles; i++){
         cellSizes[grid_host[i]]++;
    }
    cellStart_host[0] = 0;
    for (int i = 1; i <= numCells; i++)
    {
       cellStart_host[i] = cellStart_host[i-1] + cellSizes[i-1];
    }

    cudaMemcpy(d_cellStart, cellStart_host.data(), (numCells+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, grid_host.data(), params.numParticles*sizeof(int), cudaMemcpyHostToDevice);

    float *h_mass = new float[params.numParticles];
    float *h_radius = new float[params.numParticles];
    for (int i = 0; i < params.numParticles; i++){
        h_mass[i] = particles[i].mass;
        h_radius[i] = particles[i].radius;
    }

    cudaMemcpy(d_pos[0], &particles[0].position, particleMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel[0], &particles[0].velocity, particleMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, massMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_radius, h_radius, massMemSize, cudaMemcpyHostToDevice);

    // --- Simulation Loop ---
    float dt = params.maxTimeStep;
    int currentBuffer = 0; // Toggle between buffers for positions and velocities

    std::ofstream outputFile("particle_positions.txt");

    if(!outputFile.is_open())
    {
      std::cerr << "Error opening output file" << std::endl;
      return;
    }


    for (int iter = 0; iter < params.maxIterations; iter++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Calculate Forces
        int threadsPerBlock = 256;
        int blocksPerGrid = (params.numParticles + threadsPerBlock - 1) / threadsPerBlock;
        calculateForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_pos[currentBuffer], d_vel[currentBuffer], d_mass, d_radius,
                                                                d_force, d_grid, d_cellStart,
                                                                numCellsX, numCellsY, numCellsZ,
                                                                params.numParticles, params.boxSize, params.gridSize, params.gravity,
                                                                params.dragCoefficient, params.pairwiseForceStrength, params.cutoffDistance);

        cudaDeviceSynchronize();

        // Integrate
        integrateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_pos[(currentBuffer + 1) % 2], d_vel[(currentBuffer + 1) % 2], d_force, d_mass, d_radius, params.numParticles, dt, params.elasticity);
        cudaDeviceSynchronize();

        //--- Adaptive Time Stepping
        //For now, use fixed time step
       //float maxAcc = 0.0f;
       //   float* h_force = new float[params.numParticles * 3];
       //cudaMemcpy(h_force, d_force, params.numParticles * sizeof(float3), cudaMemcpyDeviceToHost);
       //    for(int i = 0; i < params.numParticles; i++){
       //       float acc =  sqrtf(h_force[3*i]* h_force[3*i] /h_mass[i] / h_mass[i] +
       //                         h_force[3*i+1]* h_force[3*i+1] / h_mass[i] / h_mass[i] +
       //                         h_force[3*i+2]* h_force[3*i+2] / h_mass[i] / h_mass[i]
       //                         );
       //     maxAcc = std::max(maxAcc, acc);
       //   }
       //  dt = std::min(std::max(params.safetyFactor * sqrtf(0.01f / maxAcc), params.minTimeStep), params.maxTimeStep);

        currentBuffer = (currentBuffer + 1) % 2;


        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Iteration: " << iter << ", time: " << duration.count() << " microseconds" << std::endl;

        if (iter % params.printInterval == 0) {
             std::vector<float3> pos(params.numParticles);
             cudaMemcpy(pos.data(), d_pos[currentBuffer], particleMemSize, cudaMemcpyDeviceToHost);
            std::cout << "Positions at Iteration: " << iter << std::endl;
            for (int i = 0; i < params.numParticles; i++) {
               std::cout << "Particle " << i << ": " << pos[i].x << ", " << pos[i].y << ", " << pos[i].z << std::endl;
                outputFile << iter << " " << pos[i].x << " " << pos[i].y << " " << pos[i].z << std::endl;
           }
        }
    }

    outputFile.close();

    // --- Cleanup ---
    cudaFree(d_pos[0]);
    cudaFree(d_pos[1]);
    cudaFree(d_vel[0]);
    cudaFree(d_vel[1]);
    cudaFree(d_mass);
    cudaFree(d_radius);
    cudaFree(d_force);
     cudaFree(d_grid);
    cudaFree(d_cellStart);
    delete [] h_mass;
    delete [] h_radius;

}


void initializeParticles(std::vector<Particle>& particles, const SimulationParams& params) {
   std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-params.boxSize/2.0f, params.boxSize/2.0f);
    std::uniform_real_distribution<float> massDist(0.5f, 1.5f); // Example mass range
    std::uniform_real_distribution<float> radiusDist(0.1f, 0.2f); // Example radius range

     particles.resize(params.numParticles);
    for(int i = 0; i < params.numParticles; ++i){
        particles[i].position = {dist(rng),dist(rng),dist(rng)};
        particles[i].velocity = {0.0f, 0.0f, 0.0f};
        particles[i].mass = massDist(rng);
        particles[i].radius = radiusDist(rng);
    }
}


void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found.\n";
        return;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << "\n";
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << "\n";
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per multi-processor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "------------------------------------------\n";
    }
}

int main() {
    printDeviceInfo();


   SimulationParams params;
    params.numParticles = 100;
    params.boxSize = 10.0f;
    params.gridSize = 1.0f;
    params.gravity = 0.1f;
    params.dragCoefficient = 0.01f;
    params.pairwiseForceStrength = 0.05f; // Linear force strength
    params.cutoffDistance = 2.0f;
    params.maxTimeStep = 0.01f;
    params.minTimeStep = 0.0001f;
    params.safetyFactor = 0.9f;
    params.elasticity = 0.7f;
    params.maxIterations = 200;
    params.printInterval = 50;

    std::vector<Particle> particles;
    initializeParticles(particles, params);
    runSimulation(particles, params);


    return 0;
}