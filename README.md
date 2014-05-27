Cuda Floyd Warshall implementation
===========================
<b>Author: Mateusz Bojanowski</b>

<hr/>
CUDA implementation of the Blocked Floyd-Warshall All pairs shortest path graph algorithm
based on article:
"A Multi-Stage CUDA Kernel for Floyd-Warshall" (Ben Lund, Justin W. Smith)

<hr/>
<b> Floyd Warshall on CUDA (16 x 16 BLOCK SIZE and 2x2 THREAD SIZE for blocked-fw-cuda) with predecessors</b>
<br/>
<b> Hardware: GTX 480 and Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz</b> 

<table>
  <tr>
    <th>|V|</th><th>|E|</th><th>fw.cpp</th><th> fw-cuda.cu </th><th>Speedup</th><th> blocked-fw-cuda.cu </th><th>Speedup</th>
  </tr>
  <tr>
    <td> 200 </td><td> 6000 </td><td> 0.0677s</td><td> 0.00158s </td><td> 42.85x</td></td><td> 0.000931s </td><td> 72.76x</td>
  </tr>
  <tr>
    <td> 400 </td><td> 24000 </td><td> 0.3519s</td><td> 0.00635s </td><td> 55.40x</td></td><td> 0.002932s </td><td> 120.00x</td>
  </tr>
  <tr>
    <td> 800</td><td> 96000 </td><td> 2.2516s</td><td> 0.03852s </td><td> 58.46x</td></td><td> 0.012195s </td><td> 184.62x</td>
  </tr>
  <tr>
    <td> 1000</td><td> 150000 </td><td> 4.3070s</td><td> 0.07766s </td><td>55.46x</td></td><td> 0.022488s </td><td> 191.52x</td>
  </tr
  <tr>
    <td> 2000</td><td> 400000 </td><td> 34.2439s</td><td> 0.5327s </td><td> 64.28</td></td><td> 0.139553s </td><td> 245.38x</td>
  </tr>
  <tr>
    <td> 5000</td><td> 500000</td><td> 541.0437s</td><td> 8.3977s </td><td> 64.43</td></td><td> 1.95655s </td><td> 276.53x</td>
  </tr>
  <tr>
    <td> 7500</td><td> 1125000</td><td> 1826.0225s</td><td> 29.0923s </td><td> 62.77</td></td><td> 6.41957s </td><td> 284.45x</td>
  </tr>
  <tr>
    <td> 10000</td><td> 2000000</td><td> 4328.3497s</td><td> 64.7613s </td><td> 66.84</td></td><td> 14.96603s </td><td> 289.21x</td>
  </tr>
  <tr>
    <td> 12500</td><td> 3125000</td><td> 8453.8079s</td><td> 132.5225s </td><td> 63.79</td></td><td> 28.89446s </td><td> 292.57x</td>
  </tr>
</table> 
