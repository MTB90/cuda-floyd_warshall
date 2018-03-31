Cuda Floyd Warshall implementation
===========================
CUDA implementation of the Blocked Floyd-Warshall All pairs shortest path graph algorithm
based on article:
"A Multi-Stage CUDA Kernel for Floyd-Warshall" (Ben Lund, Justin W. Smith)

#### Tested:
<table style="width:100%; border:0px" >
 <td>
  <b>Hardware (MSI GP72 7RE):</b>
  <ol>
   <li>Processor: Intel(R) Core(TM) i7-7700HQ</li>
   <li>GPU: GTX 1050 Ti 2GB RAM</li>
   <li>RAM: 8GB RAM</li>
  </ol> 
 </td>
 <td>
  <b>Environment:</b>
  <ol>
   <li>System: Ubuntu 17.10</li>
   <li>NVCC: 9.1</li>
   <li>CC: 6.1</li>
  </ol>
 </td>
</table>


#### Performance results:

<table>
 <tr>
   <th>|V|</th><th>|E|</th><th>fw.cpp</th><th> fw-cuda.cu </th><th>Speedup</th><th> blocked-fw-cuda.cu </th><th>Speedup</th>
 </tr>
 <tr>
  <td> 1000</td><td> 150000 </td><td> 0.724s</td><td> 0.176s </td><td> 4.215x</td></td><td> 0.117s</td><td> 6.188x</td>
 </tr>
 <tr>
   <td> 2000</td><td> 400000 </td><td> 5.895s</td><td> 0.642s </td><td> 9.182x</td></td><td> 0.213s</td><td> 27.67x</td>
 </tr>
 <tr>
   <td> 5000</td><td> 500000</td><td> 84.91s</td><td> 8.406s </td><td> 10.10x</td></td><td> 2.301s</td><td> 36.90x</td>
 </tr>
 <tr>
   <td> 7500</td><td> 1125000</td><td> 279.0s</td><td> 28.31s </td><td> 9.855x</td></td><td> 7.550s</td><td> 36.95x</td>
 </tr>
 <tr>
   <td> 10000</td><td> 2000000</td><td> 665.9s</td><td> 63.81s </td><td> 10.43x</td></td><td> 17.28s</td><td> 38.53x</td>
 </tr>
 <tr>
   <td> 12500</td><td> 3125000</td><td> 1269s</td><td> 125.5s</td><td> 10.11x</td></td><td> 34.03s</td><td> 37.27x</td>
 </tr>
</table>

#### Compile source:
<ol>
  <li>Install make/nvcc</li>
  <li>Update makefile with path to nvcc compiler</li>
  <li>Run command: make all</li>
</ol>

#### Run tests:
<ol>
  <li>Install Python 3.6</li>
  <li>Go to project directory</li>
  <li>Run command: python3 -m unittest discover -s test -v</li>
</ol>

<hr/>
<b>Author: Mateusz Bojanowski</b>
