#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants
{

    SceneName sceneName;

    int numberOfCircles;

    float *position;
    float *velocity;
    float *color;
    float *radius;
    float *posRad4; // packed as float4 array (pos.xyz, radius)

    int imageWidth;
    int imageHeight;
    float *imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int cuConstNoiseYPermutationTable[256];
__constant__ int cuConstNoiseXPermutationTable[256];
__constant__ float cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float cuConstColorRamp[COLOR_MAP_SIZE][3];

// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"

#define SCAN_BLOCK_DIM 1024
#include "exclusiveScan.cu_inl"

static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake()
{

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4 *)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a)
{

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4 *)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks()
{
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float *velocity = cuConstRendererParams.velocity;
    float *position = cuConstRendererParams.position;
    float *radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS)
    { // firework center; no update
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i + 1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j + 1] += velocity[index3j + 1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j + 1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist)
    { // restore to starting position
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j + 1] = position[index3i + 1] + y;
        position[index3j + 2] = 0.0f;

        // Travel scaled unit length
        velocity[index3j] = cosA / 5.0;
        velocity[index3j + 1] = sinA / 5.0;
        velocity[index3j + 2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float *radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff)
    {
        radius[index] = 0.02f;
    }
    else
    {
        radius[index] += 0.01f;
    }
}

// kernelAdvanceBouncingBalls
//
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls()
{
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float *velocity = cuConstRendererParams.velocity;
    float *position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3 + 1];
    float oldPosition = position[index3 + 1];

    if (oldVelocity == 0.f && oldPosition == 0.f)
    { // stop-condition
        return;
    }

    if (position[index3 + 1] < 0 && oldVelocity < 0.f)
    { // bounce ball
        velocity[index3 + 1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3 + 1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3 + 1] += velocity[index3 + 1] * dt;

    if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon && oldPosition < 0.0f && fabsf(position[index3 + 1] - oldPosition) < epsilon)
    { // stop ball
        velocity[index3 + 1] = 0.f;
        position[index3 + 1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake()
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float *positionPtr = &cuConstRendererParams.position[index3];
    float *velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3 *)positionPtr);
    float3 velocity = *((float3 *)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ((position.y + radius < 0.f) ||
        (position.x + radius) < -0.f ||
        (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3 *)positionPtr) = position;
    *((float3 *)velocityPtr) = velocity;
}

__device__ inline void atomicBlendAssign(float *addr, float alpha, float src)
{
    int *iaddr = reinterpret_cast<int *>(addr);
    int old = *iaddr;
    while (true)
    {
        float oldf = __int_as_float(old);
        float newf = alpha * src + (1.f - alpha) * oldf;
        int newi = __float_as_int(newf);
        int prev = atomicCAS(iaddr, old, newi);
        if (prev == old)
            break;
        old = prev;
    }
}

// Pack position.xyz and radius into float4 array on device
__global__ void kernelPackPosRad4(const float *position,
                                  const float *radius,
                                  int numCircles,
                                  float *posRad4)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCircles)
        return;
    int i3 = 3 * i;
    float4 pr = make_float4(position[i3 + 0], position[i3 + 1], position[i3 + 2], radius[i]);
    ((float4 *)posRad4)[i] = pr;
}

// kernelRenderPixels -- (CUDA device code)
//
// Each thread shades one pixel by looping over all circles in input order.
// This preserves per-pixel ordering and avoids atomics because a single
// thread owns the pixel's full read-modify-write sequence.
__global__ void kernelRenderPixels()
{

    int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (offsetX >= width || offsetY >= height)
        return;

    int offset = 4 * (offsetY * width + offsetX);

    // read current pixel color
    float4 pixelColor = *(float4 *)(&cuConstRendererParams.imageData[offset]);
    float r = pixelColor.x;
    float g = pixelColor.y;
    float b = pixelColor.z;
    float a = pixelColor.w;

    // Pixel center in normalized coordinates
    float invWidth = 1.f / width;
    float invHeight = 1.f / height;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(offsetX) + 0.5f),
                                         invHeight * (static_cast<float>(offsetY) + 0.5f));

    // for every circle that contain this pixel
    int numCircles = cuConstRendererParams.numberOfCircles;
    for (int i = 0; i < numCircles; i++)
    {
        float4 pr = *(float4 *)(&cuConstRendererParams.posRad4[4 * i]);
        float rad = pr.w;

        float diffX = pr.x - pixelCenterNorm.x;
        float diffY = pr.y - pixelCenterNorm.y;
        float pixelDist = diffX * diffX + diffY * diffY;
        float maxDist = rad * rad;

        // Circle does not contribute to the image
        if (pixelDist > maxDist)
            continue;

        // Shade: compute rgb and alpha for this circle at this pixel
        float3 rgb;
        float alpha;
        if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME)
        {
            const float kCircleMaxAlpha = .5f;
            const float falloffScale = 4.f;
            float normPixelDist = sqrtf(pixelDist) / rad;
            rgb = lookupColor(normPixelDist);
            float maxAlpha = .6f + .4f * (1.f - pr.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * expf(-1.f * falloffScale * normPixelDist * normPixelDist);
        }
        else
        {
            int index3 = 3 * i;
            rgb = *(float3 *)&(cuConstRendererParams.color[index3]);
            alpha = .5f;
        }

        // In-order blend into this pixel (local registers)
        float oneMinus = 1.f - alpha;
        r = alpha * rgb.x + oneMinus * r;
        g = alpha * rgb.y + oneMinus * g;
        b = alpha * rgb.z + oneMinus * b;
        a = a + alpha;
    }

    *(float4 *)(&cuConstRendererParams.imageData[offset]) = make_float4(r, g, b, a);
}

////////////////////////////////////////////////////////////////////////////////////////
// GPU binning: count -> host scan -> fill -> per-tile serial sort
////////////////////////////////////////////////////////////////////////////////////////

// Count tiles overlapped per circle (precise circle-box intersection)
__global__ void kernelCountTilesPerCircle(int tilesX,
                                          int tilesY,
                                          int tileW,
                                          int tileH,
                                          int *circleCounts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int numCircles = cuConstRendererParams.numberOfCircles;
    if (i >= numCircles)
        return;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    int index4 = 4 * i;
    float4 pr = *(float4 *)(&cuConstRendererParams.posRad4[index4]);

    float cx = pr.x;
    float cy = pr.y;
    float radius = pr.w;

    // Convert to normalized coordinates for tile calculations
    // We use normalized coordinates (in [0,1]) so that all geometric calculations are independent of the actual image resolution.
    // This allows us to easily compare circle and tile positions and sizes, regardless of pixel dimensions.
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float tileW_norm = tileW * invWidth;
    float tileH_norm = tileH * invHeight;

    int count = 0;

    // Test each tile for intersection with the circle
    for (int ty = 0; ty < tilesY; ty++)
    {
        for (int tx = 0; tx < tilesX; tx++)
        {
            // Convert tile to normalized coordinates
            float tileL = tx * tileW_norm;
            float tileR = (tx + 1) * tileW_norm;
            float tileB = ty * tileH_norm;
            float tileT = (ty + 1) * tileH_norm;

            // Test if circle intersects this tile using precise intersection test
            if (circleInBox(cx, cy, radius, tileL, tileR, tileT, tileB))
            {
                count++;
            }
        }
    }

    circleCounts[i] = count;
}

// Write pairs (tileId, circleIdx) for each circle into a contiguous segment
__global__ void kernelWriteCircleTilePairs(const float *position,
                                           const float *radius,
                                           int numCircles,
                                           int imageWidth,
                                           int imageHeight,
                                           int tilesX,
                                           int tilesY,
                                           int tileW,
                                           int tileH,
                                           const int *circleBase, // An array where circleBase[i] gives the starting index in the output arrays for the i-th circle's tile pairs.
                                           int *pairTileId,       // Output array to store the tile ID for each (tile, circle) pair that the circle overlaps.
                                           int *pairCircleIdx)    // Output array to store the circle index for each (tile, circle) pair; aligns with pairTileId.
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCircles)
        return;

    float cx = position[3 * i + 0];
    float cy = position[3 * i + 1];
    float r = radius[i];

    // Convert to normalized coordinates for tile calculations
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float tileW_norm = tileW * invWidth;
    float tileH_norm = tileH * invHeight;

    int base = circleBase[i];
    int pairOffset = 0;

    // Test each tile for intersection with the circle
    for (int ty = 0; ty < tilesY; ty++)
    {
        for (int tx = 0; tx < tilesX; tx++)
        {
            // Convert tile to normalized coordinates
            float tileL = tx * tileW_norm;
            float tileR = (tx + 1) * tileW_norm;
            float tileB = ty * tileH_norm;
            float tileT = (ty + 1) * tileH_norm;

            // Test if circle intersects this tile using precise intersection test
            if (circleInBox(cx, cy, r, tileL, tileR, tileT, tileB))
            {
                int tileId = ty * tilesX + tx;
                int pairIdx = base + pairOffset;
                pairTileId[pairIdx] = tileId;
                pairCircleIdx[pairIdx] = i;
                pairOffset++;
            }
        }
    }
}

// Histogram tiles from pairs (one thread per pair entry)
__global__ void kernelHistogramTilesFromPairs(const int *pairTileId,
                                              int totalPairs, // The total number of (tile, circle) pairs
                                              int numTiles,
                                              int *tileCounts)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= totalPairs)
        return;
    int tileId = pairTileId[k];
    if (tileId >= 0 && tileId < numTiles)
        atomicAdd(&tileCounts[tileId], 1);
}

// Device-side stable counting sort by tiles using "waves" (waves are contiguous blocks of K entries, i.e., each wave processes up to waveSize pairs)
// 1) Count tiles per wave into waveCounts[m][tile]
//    - Here, a "wave" refers to a chunk of up to waveSize consecutive (tile, circle) pairs.
//    - Each block processes one wave (i.e., a range of indices [m*waveSize, (m+1)*waveSize)), allowing the sort to be performed in manageable batches.
__global__ void kernelCountTilesPerWave(const int *pairTileId,
                                        int totalPairs,
                                        int numTiles,
                                        int waveSize,
                                        int *waveCounts)
{
    int m = blockIdx.x;
    int waveStart = m * waveSize;
    if (waveStart >= totalPairs)
        return;
    int waveEnd = min(waveStart + waveSize, totalPairs);

    extern __shared__ int sCounts[]; // size = numTiles
    for (int t = threadIdx.x; t < numTiles; t += blockDim.x)
        sCounts[t] = 0;
    __syncthreads();

    for (int k = waveStart + threadIdx.x; k < waveEnd; k += blockDim.x)
    {
        int tile = pairTileId[k];
        atomicAdd(&sCounts[tile], 1);
    }
    __syncthreads();

    for (int t = threadIdx.x; t < numTiles; t += blockDim.x)
        waveCounts[m * numTiles + t] = sCounts[t];
}

// 2) Exclusive scan across waves per tile: waveBase[m][tile] = sum_{w<m} waveCounts[w][tile]
__global__ void kernelExclusiveScanWaveCounts(const int *waveCounts,
                                              int numWaves,
                                              int numTiles,
                                              int *waveBase)
{
    int tile = blockIdx.x;
    if (tile >= numTiles)
        return;
    int acc = 0;
    for (int m = 0; m < numWaves; m++)
    {
        int idx = m * numTiles + tile;
        int c = waveCounts[idx];
        waveBase[idx] = acc;
        acc += c;
    }
}

// 2) Exclusive scan across waves per tile using shared memory scan for better performance
__global__ void kernelExclusiveScanWaveCountsSharedMem(const int *waveCounts,
                                              int numWaves,
                                              int numTiles,
                                              int *waveBase,
                                              int alignedWaves)
{
    int tile = blockIdx.x;
    if (tile >= numTiles)
        return;

    // Use shared memory for parallel exclusive scan
    extern __shared__ uint sData[];
    uint *sInput = sData;
    uint *sOutput = sData + alignedWaves;
    uint *sScratch = sData + 2 * alignedWaves;

    int tid = threadIdx.x;

    // Initialize shared memory arrays
    if (tid < alignedWaves) {
        sInput[tid] = 0;
        sOutput[tid] = 0;
    }
    __syncthreads();

    // Load wave counts for this tile into shared memory
    // Only load up to numWaves, pad the rest with zeros
    if (tid < numWaves) {
        int idx = tid * numTiles + tile;
        sInput[tid] = (uint)waveCounts[idx];
    }
    __syncthreads();

    // Perform exclusive scan using shared memory
    sharedMemExclusiveScan(tid, sInput, sOutput, sScratch, alignedWaves);
    __syncthreads();

    // Write results back to global memory
    if (tid < numWaves) {
        int idx = tid * numTiles + tile;
        waveBase[idx] = (int)sOutput[tid];
    }
}

// 3) Stable scatter within each wave using wave-local heads, preserving pair order
__global__ void kernelScatterWaveStable(const int *pairTileId,
                                        const int *pairCircleIdx,
                                        int totalPairs,
                                        int waveSize,
                                        int numTiles,
                                        const int *tileOffsets,
                                        const int *waveBase,
                                        int *tileIndices)
{
    int m = blockIdx.x;
    int waveStart = m * waveSize;
    if (waveStart >= totalPairs)
        return;
    int waveEnd = min(waveStart + waveSize, totalPairs);

    extern __shared__ int heads[]; // size = numTiles
    for (int t = threadIdx.x; t < numTiles; t += blockDim.x)
        heads[t] = 0;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int k = waveStart; k < waveEnd; k++)
        {
            int tile = pairTileId[k];
            int pos = tileOffsets[tile] + waveBase[m * numTiles + tile] + heads[tile];
            tileIndices[pos] = pairCircleIdx[k];
            heads[tile]++;
        }
    }
}

// Render using pre-built CSR bins; preserves order via per-tile sorted indices
#ifndef BIN_CHUNK
#define BIN_CHUNK 128
#endif
#ifndef DIRECT_SEG_LIMIT
#define DIRECT_SEG_LIMIT 64
#endif
__global__ void kernelRenderPixelsBinned(const int *tileOffsets,
                                         const int *tileIndices,
                                         int tilesNumX,
                                         int tileW,
                                         int tileH)
{

    // Map one CUDA block to one tile: threads cover the tile's pixels
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    int tileX = blockIdx.x;
    int tileY = blockIdx.y;

    int offsetX = tileX * tileW + threadIdx.x;
    int offsetY = tileY * tileH + threadIdx.y;

    bool inBounds = (offsetX < width) && (offsetY < height);
    int pixelOffset = 0;
    float r = 0.f, g = 0.f, b = 0.f, a = 0.f;
    // no early-return
    if (inBounds)
    {
        pixelOffset = 4 * (offsetY * width + offsetX);
        float4 pixelColor = *(float4 *)(&cuConstRendererParams.imageData[pixelOffset]);
        r = pixelColor.x;
        g = pixelColor.y;
        b = pixelColor.z;
        a = pixelColor.w;
    }

    float invWidth = 1.f / width;
    float invHeight = 1.f / height;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(offsetX) + 0.5f),
                                         invHeight * (static_cast<float>(offsetY) + 0.5f));

    int tileId = tileY * tilesNumX + tileX;
    // tileOffsets is a CSR (Compressed Sparse Row) array of length (numTiles + 1).
    // For each tileId, tileOffsets[tileId] gives the starting index in the tileIndices array
    // for the circles that overlap this tile. tileOffsets[tileId+1] is the end index (exclusive).
    int begin = tileOffsets[tileId];
    int end = tileOffsets[tileId + 1];
    int segLen = end - begin; // how many circles in this tile

    // Fast path for small segments: avoid shared-memory chunking overhead
    if (segLen <= DIRECT_SEG_LIMIT)
    {
        for (int k = begin; k < end; k++)
        {
            int i = tileIndices[k];
            float4 pr = *((float4 *)(&cuConstRendererParams.posRad4[4 * i]));
            float rad = pr.w;

            float diffX = pr.x - pixelCenterNorm.x;
            float diffY = pr.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            float maxDist = rad * rad;
            if (pixelDist > maxDist)
                continue;

            float3 rgb;
            float alpha;
            if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME)
            {
                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;
                float normPixelDist = sqrtf(pixelDist) / rad;
                rgb = lookupColor(normPixelDist);
                float maxAlpha = .6f + .4f * (1.f - pr.z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                alpha = maxAlpha * expf(-1.f * falloffScale * normPixelDist * normPixelDist);
            }
            else
            {
                int index3 = 3 * i;
                rgb = *(float3 *)&(cuConstRendererParams.color[index3]);
                alpha = .5f;
            }

            float oneMinus = 1.f - alpha;
            r = alpha * rgb.x + oneMinus * r;
            g = alpha * rgb.y + oneMinus * g;
            b = alpha * rgb.z + oneMinus * b;
            a = a + alpha;
        }
        if (inBounds)
            *(float4 *)(&cuConstRendererParams.imageData[pixelOffset]) = make_float4(r, g, b, a);
        return;
    }

    __shared__ float4 sPR[BIN_CHUNK];
    __shared__ int sIdx[BIN_CHUNK];

    int linearThread = threadIdx.y * blockDim.x + threadIdx.x;
    int threadsPerBlock = blockDim.x * blockDim.y;

    for (int base = begin; base < end; base += BIN_CHUNK)
    {
        int chunkLen = min(BIN_CHUNK, end - base);

        // Cooperative load of circle parameters into shared memory
        for (int t = linearThread; t < chunkLen; t += threadsPerBlock)
        {
            int i = tileIndices[base + t];
            sIdx[t] = i;
            sPR[t] = *((float4 *)(&cuConstRendererParams.posRad4[4 * i]));
        }
        __syncthreads();

        // Shade against the shared chunk
        for (int j = 0; j < chunkLen; j++)
        {
            float4 pr = sPR[j];
            float rad = pr.w;

            float diffX = pr.x - pixelCenterNorm.x;
            float diffY = pr.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            float maxDist = rad * rad;
            if (pixelDist > maxDist)
                continue;

            float3 rgb;
            float alpha;
            if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME)
            {
                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;
                float normPixelDist = sqrtf(pixelDist) / rad;
                rgb = lookupColor(normPixelDist);
                float maxAlpha = .6f + .4f * (1.f - pr.z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                alpha = maxAlpha * expf(-1.f * falloffScale * normPixelDist * normPixelDist);
            }
            else
            {
                int i = sIdx[j];
                int index3 = 3 * i;
                rgb = *(float3 *)&(cuConstRendererParams.color[index3]);
                alpha = .5f;
            }

            float oneMinus = 1.f - alpha;
            r = alpha * rgb.x + oneMinus * r;
            g = alpha * rgb.y + oneMinus * g;
            b = alpha * rgb.z + oneMinus * b;
            a = a + alpha;
        }
        __syncthreads();
    }

    if (inBounds)
        *(float4 *)(&cuConstRendererParams.imageData[pixelOffset]) = make_float4(r, g, b, a);
}

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer()
{
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
    cudaDevicePosRad4 = NULL;
}

CudaRenderer::~CudaRenderer()
{

    if (image)
    {
        delete image;
    }

    if (position)
    {
        delete[] position;
        delete[] velocity;
        delete[] color;
        delete[] radius;
    }

    if (cudaDevicePosition)
    {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        if (cudaDevicePosRad4)
            cudaFree(cudaDevicePosRad4);
    }
}

const Image *
CudaRenderer::getImage()
{

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void CudaRenderer::loadScene(SceneName scene)
{
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void CudaRenderer::setup()
{

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
    cudaMalloc(&cudaDevicePosRad4, sizeof(float) * 4 * numberOfCircles);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);
    // Pack pos.xyz + radius into float4 array on device
    {
        dim3 block(256, 1, 1);
        dim3 grid((numberOfCircles + block.x - 1) / block.x, 1, 1);
        kernelPackPosRad4<<<grid, block>>>(cudaDevicePosition, cudaDeviceRadius, numberOfCircles, cudaDevicePosRad4);
    }

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.posRad4 = cudaDevicePosRad4;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int *permX;
    int *permY;
    float *value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void CudaRenderer::allocOutputImage(int width, int height)
{

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void CudaRenderer::clearImage()
{

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME)
    {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    }
    else
    {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void CudaRenderer::advanceAnimation()
{
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES)
    {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    }
    else if (sceneName == BOUNCING_BALLS)
    {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    }
    else if (sceneName == HYPNOSIS)
    {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    }
    else if (sceneName == FIREWORKS)
    {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();

    // Re-pack pos.xyz + radius into float4 each frame after animation updates
    {
        dim3 packBlock(256, 1, 1);
        dim3 packGrid((numberOfCircles + packBlock.x - 1) / packBlock.x, 1, 1);
        kernelPackPosRad4<<<packGrid, packBlock>>>(cudaDevicePosition, cudaDeviceRadius, numberOfCircles, cudaDevicePosRad4);
        cudaDeviceSynchronize();
    }
}

void CudaRenderer::render()
{
    // Small-N fallback: direct per-pixel rendering avoids overhead
    if (numberOfCircles <= 1024)
    {
        dim3 blockDim(32, 32, 1);
        dim3 gridDim(
            (image->width + blockDim.x - 1) / blockDim.x,
            (image->height + blockDim.y - 1) / blockDim.y);
        kernelRenderPixels<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        return;
    }

    // Tiled, order-correct rendering using CSR bins (built each frame)

    // 1) Compute tile grid (align block=tile; keep <=1024 threads per block)
    int tileW = 32;
    int tileH = 32;
    const int tilesX = (image->width + tileW - 1) / tileW;
    const int tilesY = (image->height + tileH - 1) / tileH;
    const int numTiles = tilesX * tilesY;

    // 2) Per-circle counts of overlapped tiles (device)
    int *dCircleCounts = NULL;
    cudaMalloc(&dCircleCounts, sizeof(int) * numberOfCircles);
    dim3 blockCount(256, 1, 1);
    dim3 gridCount((numberOfCircles + blockCount.x - 1) / blockCount.x, 1, 1);
    kernelCountTilesPerCircle<<<gridCount, blockCount>>>(tilesX, tilesY, tileW, tileH, dCircleCounts);

    // 3) Host exclusive scan over per-circle counts -> circleBase, totalPairs
    std::vector<int> hCircleCounts(numberOfCircles);
    cudaMemcpy(hCircleCounts.data(), dCircleCounts, sizeof(int) * numberOfCircles, cudaMemcpyDeviceToHost);
    std::vector<int> hCircleBase(numberOfCircles + 1);
    hCircleBase[0] = 0;
    for (int i = 0; i < numberOfCircles; i++)
        hCircleBase[i + 1] = hCircleBase[i] + hCircleCounts[i];
    int totalPairs = hCircleBase[numberOfCircles];

    int *dCircleBase = NULL;
    cudaMalloc(&dCircleBase, sizeof(int) * numberOfCircles);
    cudaMemcpy(dCircleBase, hCircleBase.data(), sizeof(int) * numberOfCircles, cudaMemcpyHostToDevice);

    // 4) Build pairs (tileId, circleIdx) by circle (device, ordered by circle)
    int *dPairTileId = NULL;
    int *dPairCircleIdx = NULL;
    cudaMalloc(&dPairTileId, sizeof(int) * totalPairs);
    cudaMalloc(&dPairCircleIdx, sizeof(int) * totalPairs);
    kernelWriteCircleTilePairs<<<gridCount, blockCount>>>(
        cudaDevicePosition, cudaDeviceRadius, numberOfCircles,
        image->width, image->height,
        tilesX, tilesY, tileW, tileH,
        dCircleBase, dPairTileId, dPairCircleIdx);

    // 5) Histogram tiles from pairs -> counts (device); host scan -> CSR offsets
    int *dTileCounts = NULL;
    cudaMalloc(&dTileCounts, sizeof(int) * numTiles);
    cudaMemset(dTileCounts, 0, sizeof(int) * numTiles);
    dim3 blockPairs(256, 1, 1);
    dim3 gridPairs((totalPairs + blockPairs.x - 1) / blockPairs.x, 1, 1);
    kernelHistogramTilesFromPairs<<<gridPairs, blockPairs>>>(dPairTileId, totalPairs, numTiles, dTileCounts);

    std::vector<int> hCounts(numTiles);
    cudaMemcpy(hCounts.data(), dTileCounts, sizeof(int) * numTiles, cudaMemcpyDeviceToHost);
    std::vector<int> hOffsets(numTiles + 1);
    hOffsets[0] = 0;
    for (int t = 0; t < numTiles; t++)
        hOffsets[t + 1] = hOffsets[t] + hCounts[t];

    int *dTileOffsets = NULL;
    cudaMalloc(&dTileOffsets, sizeof(int) * (numTiles + 1));
    cudaMemcpy(dTileOffsets, hOffsets.data(), sizeof(int) * (numTiles + 1), cudaMemcpyHostToDevice);

    // 6) Stable counting sort by tiles
    int *dTileIndices = NULL;
    cudaMalloc(&dTileIndices, sizeof(int) * totalPairs);

    const int pairCutoff = 2000;
    printf("totalPairs: %d\n", totalPairs);
    if (totalPairs <= pairCutoff)
    {
        // Host-side stable scatter (only for very small sizes)
        std::vector<int> hPairTileId(totalPairs);
        std::vector<int> hPairCircleIdx(totalPairs);
        cudaMemcpy(hPairTileId.data(), dPairTileId, sizeof(int) * totalPairs, cudaMemcpyDeviceToHost);
        cudaMemcpy(hPairCircleIdx.data(), dPairCircleIdx, sizeof(int) * totalPairs, cudaMemcpyDeviceToHost);

        std::vector<int> hTileIndices(totalPairs);
        std::vector<int> heads(numTiles);
        for (int t = 0; t < numTiles; t++)
            heads[t] = hOffsets[t];
        for (int k = 0; k < totalPairs; k++)
        {
            int t = hPairTileId[k];
            int pos = heads[t]++;
            hTileIndices[pos] = hPairCircleIdx[k];
        }
        cudaMemcpy(dTileIndices, hTileIndices.data(), sizeof(int) * totalPairs, cudaMemcpyHostToDevice);
    }
    else
    {
        // Optimized wave sizing for medium-sized workloads (2000-10000 pairs)
        int targetWaves;
        if (totalPairs <= 10000) {
            targetWaves = 64;
        } else if (totalPairs <= 200000) {
            targetWaves = 256;
        } else {
            targetWaves = 512; // Large waves for big workloads
        }
        
        int waveSize = max(64, (totalPairs + targetWaves - 1) / targetWaves); 
        int numWaves = (totalPairs + waveSize - 1) / waveSize;
        
        printf("Adaptive wave sizing, waveSize=%d, numWaves=%d\n", waveSize, numWaves);
        int *dWaveCounts = NULL;
        int *dWaveBase = NULL;
        cudaMalloc(&dWaveCounts, sizeof(int) * numWaves * numTiles);
        cudaMalloc(&dWaveBase, sizeof(int) * numWaves * numTiles);

        // Count per wave
        int shmemCounts = sizeof(int) * numTiles;
        kernelCountTilesPerWave<<<numWaves, 256, shmemCounts>>>(dPairTileId, totalPairs, numTiles, waveSize, dWaveCounts);

        // Exclusive scan across waves per tile
        printf("totalPairs: %d, numTiles: %d, numWaves: %d\n", totalPairs, numTiles, numWaves);
        
        // Adaptive strategy: choose optimal scan method based on workload
        int alignedWaves = nextPow2(numWaves);
        alignedWaves = min(alignedWaves, SCAN_BLOCK_DIM);
        
        float efficiency = 100.0 * numWaves / alignedWaves;
        printf("Scan analysis: numWaves=%d, alignedWaves=%d, efficiency=%.1f%%\n", numWaves, alignedWaves, efficiency);
        
        if (numWaves >= 64 && efficiency >= 50.0) {
            int shmemScanSize = sizeof(uint) * (alignedWaves + alignedWaves + 2 * SCAN_BLOCK_DIM);
            kernelExclusiveScanWaveCountsSharedMem<<<numTiles, alignedWaves, shmemScanSize>>>(dWaveCounts, numWaves, numTiles, dWaveBase, alignedWaves);
        } else {
            kernelExclusiveScanWaveCounts<<<numTiles, 1>>>(dWaveCounts, numWaves, numTiles, dWaveBase);
        }

        // Scatter stably within each wave into final CSR locations
        int shmemHeads = sizeof(int) * numTiles;
        kernelScatterWaveStable<<<numWaves, 256, shmemHeads>>>(dPairTileId, dPairCircleIdx, totalPairs,
                                                               waveSize, numTiles, dTileOffsets, dWaveBase, dTileIndices);

        cudaFree(dWaveCounts);
        cudaFree(dWaveBase);
    }

    // 7) Render per pixel binning (one block per tile, one thread per pixel)
    dim3 blockRender(tileW, tileH, 1);
    dim3 gridRender(tilesX, tilesY);
    kernelRenderPixelsBinned<<<gridRender, blockRender>>>(dTileOffsets, dTileIndices, tilesX, tileW, tileH);

    // 8) Cleanup temporaries
    cudaFree(dCircleCounts);
    cudaFree(dCircleBase);
    cudaFree(dPairTileId);
    cudaFree(dPairCircleIdx);
    cudaFree(dTileCounts);
    cudaFree(dTileOffsets);
    cudaFree(dTileIndices);
}
