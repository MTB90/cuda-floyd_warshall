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
  <td> 1000</td><td> 150000 </td><td> 0.724s</td><td> 0.081s </td><td>8.938x</td></td><td> 0.017s </td><td> 42.59x</td>
 </tr>
 <tr>
   <td> 2000</td><td> 400000 </td><td> 5.895s</td><td> 0.560s </td><td> 10.52x</td></td><td> 0.092s </td><td> 64.07x</td>
 </tr>
 <tr>
   <td> 5000</td><td> 500000</td><td> 84.91s</td><td> 8.302s </td><td> 10.22x</td></td><td> 1.258s </td><td> 67.49x</td>
 </tr>
 <tr>
   <td> 7500</td><td> 1125000</td><td> 279.0s</td><td> 27.36s </td><td> 10.19x</td></td><td> 4.098s </td><td> 68.08x</td>
 </tr>
 <tr>
   <td> 10000</td><td> 2000000</td><td> 665.9s</td><td> 63.94s </td><td> 10.41x</td></td><td> 9.610s</td><td> 69.29x</td>
 </tr>
 <tr>
   <td> 12500</td><td> 3125000</td><td> 1269s</td><td> 125.1s</td><td> 10.14x</td></td><td> 18.27s</td><td> 69.45x</td>
 </tr>
</table>

<hr/>
<b>Author: Mateusz Bojanowski</b>
