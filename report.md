# Report

## 1

```shell
➜  saxpy git:(problem2) ✗ ./cudaSaxpy 
---------------------------------------------------------
Found 1 CUDA devices
Device 0: NVIDIA GeForce RTX 2080
   SMs:        46
   Global mem: 7959 MB
   CUDA Cap:   7.5
---------------------------------------------------------
Kernel: 0.696 ms                1.862 %
Overall: 37.392 ms              [5.978 GB/s]
Kernel: 0.687 ms                1.698 %
Overall: 40.492 ms              [5.520 GB/s]
Kernel: 0.684 ms                1.690 %
Overall: 40.493 ms              [5.520 GB/s]
➜  saxpy git:(problem2) ✗ 
```

## 2

```shell
-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Target Time     | Your Time       | Score           |
-------------------------------------------------------------------------
| 10000           | 0.246           | 0.063           | 1.25            |
| 100000          | 0.273           | 0.118           | 1.25            |
| 1000000         | 0.467           | 0.287           | 1.25            |
| 2000000         | 0.867           | 0.683           | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5/5             |
-------------------------------------------------------------------------

-------------------------
Find_peaks Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Target Time     | Your Time       | Score           |
-------------------------------------------------------------------------
| 10000           | 0.206           | 0.094           | 1.25            |
| 100000          | 0.298           | 0.146           | 1.25            |
| 1000000         | 0.578           | 0.560           | 1.25            |
| 2000000         | 0.995           | 1.109           | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5/5             |
-------------------------------------------------------------------------

```

## 3

```shell
------------
Score table:
------------
-------------------------------------------------------------------------
| Scene Name      | Target Time     | Your Time       | Score           |
-------------------------------------------------------------------------
| rgb             | 0.1932          | 0.1820          | 12              |
| rand10k         | 1.9604          | 1.7923          | 12              |
| rand100k        | 18.5644         | 14.3695         | 12              |
| pattern         | 0.2776          | 0.3290          | 12              |
| snowsingle      | 7.7362          | 1.2804          | 12              |
| biglittle       | 14.2586         | 15.5748         | 12              |
-------------------------------------------------------------------------
|                                   | Total score:    | 72/72           |
```

rgb:
rand10k:
    totalPairs: 182,851
    Adaptive wave sizing, waveSize=1429, numWaves=128
    Scan analysis: numWaves=128, alignedWaves=128, efficiency=100.0%
rand100k:
    totalPairs: 1,830,826
    Adaptive wave sizing, waveSize=3576, numWaves=512
    Scan analysis: numWaves=512, alignedWaves=512, efficiency=100.0%
pattern:
    totalPairs: 4835
snowsingle:
    totalPairs: 67,810
    Adaptive wave sizing, waveSize=530, numWaves=128
    Scan analysis: numWaves=128, alignedWaves=128, efficiency=100.0%
biglittle:
    totalPairs: 2,075,721
    Adaptive wave sizing, waveSize=4055, numWaves=512
    Scan analysis: numWaves=512, alignedWaves=512, efficiency=100.0%

```shell
Running benchmark, 1 frames, beginning at frame 0 ...
Circle base time: 0.064 ms
Build pairs time: 0.073 ms
Histogram tiles from pairs time: 0.014 ms
Host offsets time: 0.021 ms
totalPairs: 4835, tilesX: 36, tilesY: 36
Stable sort time: 0.041 ms
Render time: 0.135 ms
Copying image data from device
***************** Correctness check passed **************************

>>>>Circle base and Build pairs time optimized<<<<

➜  render git:(problem3) ✗ ./render --check pattern
Circle base time: 0.031 ms
Build pairs time: 0.028 ms
Histogram tiles from pairs time: 0.013 ms
Host offsets time: 0.022 ms
totalPairs: 4835, tilesX: 36, tilesY: 36
Stable sort time: 0.044 ms
Render time: 0.141 ms
Copying image data from device
***************** Correctness check passed **************************
Clear:    0.0956 ms
Advance:  0.0083 ms
Render:   0.3651 ms
Total:    0.4690 ms

Overall:  0.0291 sec (note units are seconds)
```

```shell
>>>>Profiling with Nsight Compute<<<<
kernelScatterBatchStable(const int *, const int *, int, int, int, const int *, const int *, int *), 2025-Sep-22 16:14:17, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes.sum                                                                  Mbyte                          75.60
    dram__throughput.avg.pct_of_peak_sustained_elapsed                                   %                           9.27
    lts__t_sectors_hit_rate.pct                                                                                   (!) n/a
    lts__throughput.avg.pct_of_peak_sustained_elapsed                                    %                           5.94
    sm__warps_active.avg.pct_of_peak_sustained_active                                    %                          14.16
    ---------------------------------------------------------------------- --------------- ------------------------------

kernelCountTilesPerCircle(int, int, int, int, int *), 2025-Sep-22 15:57:43, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes.sum                                                                  Kbyte                         160.42
    dram__throughput.avg.pct_of_peak_sustained_elapsed                                   %                           2.59
    lts__t_sectors_hit_rate.pct                                                                                   (!) n/a
    lts__throughput.avg.pct_of_peak_sustained_elapsed                                    %                           1.09
    sm__warps_active.avg.pct_of_peak_sustained_active                                    %                          24.41
    ---------------------------------------------------------------------- --------------- ------------------------------

kernelWriteCircleTilePairs(int, int, int, int, const int *, int *, int *), 2025-Sep-22 15:57:43, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes.sum                                                                  Mbyte                          28.24
    dram__throughput.avg.pct_of_peak_sustained_elapsed                                   %                          17.08
    lts__t_sectors_hit_rate.pct                                                                                   (!) n/a
    lts__throughput.avg.pct_of_peak_sustained_elapsed                                    %                          47.34
    sm__warps_active.avg.pct_of_peak_sustained_active                                    %                          24.84
    ---------------------------------------------------------------------- --------------- ------------------------------
```
