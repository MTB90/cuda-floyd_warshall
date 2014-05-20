CUDA_Blocked_Floyd-Warshall
===========================

CUDA implementation of the Blocked Floyd-Warshall All pairs shortest path graph algorithm
based on article:
"A Multi-Stage CUDA Kernel for Floyd-Warshall" (Ben Lund, Justin W. Smith)


<hr/>

<b> Floyd Warshall on CUDA  with predecessors</b>
<br/>
<b> Hardware: GTX 480 and Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz</b> 
<br/>
<b> 16 x 16 Threads per block </b>
<table>
  <tr>
    <th>|V|</th><th>|E|</th><th>fw.cpp</th><th> fw-cuda.cu </th><th>Speedup</th><th> blocked-fw-cuda.cu </th><th>Speedup</th>
  </tr>
  <tr>
    <td> 200 </td><td> 6000 </td><td> 0.0677s</td><td> 0.00158s </td><td> 42.85x</td></td><td> 0.000950s </td><td> 71.31x</td>
  </tr>
  <tr>
    <td> 400 </td><td> 24000 </td><td> 0.3519s</td><td> 0.00635s </td><td> 55.40x</td></td><td> 0.002969s </td><td> 118.51x</td>
  </tr>
  <tr>
    <td> 800</td><td> 96000 </td><td> 2.2516s</td><td> 0.03852s </td><td> 58.46x</td></td><td> 0.012670s </td><td> 177.71x</td>
  </tr>
  <tr>
    <td> 1000</td><td> 150000 </td><td> 4.3070s</td><td> 0.07766s </td><td>55.46x</td></td><td> 0.023078s </td><td> 186.63x</td>
  </tr
  <tr>
    <td> 2000</td><td> 400000 </td><td> 34.2439s</td><td> 0.5327s </td><td> 64.28</td></td><td> 0.142524s </td><td> 240.27x</td>
  </tr>
  <tr>
    <td> 5000</td><td> 500000</td><td> 541.0437s</td><td> 8.3977s </td><td> 64.43</td></td><td> 2.000566s </td><td> 270.45x</td>
  </tr>
  <tr>
    <td> 7500</td><td> 1125000</td><td> 1826.0225s</td><td> 29.0923s </td><td> 62.77</td></td><td> 6.585984s </td><td> 277.26x</td>
  </tr>
  <tr>
    <td> 10000</td><td> 2000000</td><td> 4328.3497s</td><td> 64.7613s </td><td> 66.84</td></td><td> 15.42630s </td><td> 280.58x</td>
  </tr>
  <tr>
    <td> 12500</td><td> 3125000</td><td> 8453.8079s</td><td> 132.5225s </td><td> 63.79</td></td><td> 29.92320s </td><td> 282.51x</td>
  </tr>
</table> 
