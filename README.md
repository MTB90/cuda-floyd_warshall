Cuda Floyd Warshall implementation
===========================
CUDA implementation of the Blocked Floyd-Warshall All pairs shortest path graph algorithm
based on article:
"A Multi-Stage CUDA Kernel for Floyd-Warshall" (Ben Lund, Justin W. Smith)

<hr/>

<table style="width:100%; border:0px solid" >
 <td>
  <b>Tested hardware (MSI GP72 7RE):</b>
  <ol>
   <li>Processor: Intel(R) Core(TM) i7-7700HQ</li>
   <li>GPU: GTX 1050 Ti 2GB RAM</li>
   <li>RAM: 8GB RAM</li>
  </ol> 
 </td>
 <td>
  <b>Testing environment:</b>
  <ol>
   <li>System: Ubuntu 17.10</li>
   <li>NVCC: 9.1</li>
   <li>CC: 6.1</li>
  </ol>
 </td>
</table>

<b>Performance resutls:</b>
<table>
 <tr>
   <th>|V|</th><th>|E|</th><th>fw.cpp</th><th> fw-cuda.cu </th><th>Speedup</th><th> blocked-fw-cuda.cu </th><th>Speedup</th>
 </tr>
 <tr>
  <td> 1000</td><td> 150000 </td><td> 0.954s</td><td> 0.086s </td><td>11.09x</td></td><td> 0.018s </td><td> 53.00x</td>
 </tr>
 <tr>
   <td> 2000</td><td> 400000 </td><td> 7.539s</td><td> 0.563s </td><td> 13.39x</td></td><td> 0.096s </td><td> 78.53x</td>
 </tr>
 <tr>
   <td> 5000</td><td> 500000</td><td> 108.9s</td><td> 8.325s </td><td> 13.08x</td></td><td> 1.268s </td><td> 85.88x</td>
 </tr>
 <tr>
   <td> 7500</td><td> 1125000</td><td> 359.3s</td><td> 27.46s </td><td> 13.08x</td></td><td> 4.136s </td><td> 86.87x</td>
 </tr>
 <tr>
   <td> 10000</td><td> 2000000</td><td> 839.0s</td><td> 64.24s </td><td> 13.06x</td></td><td> 9.589s</td><td> 87.49x</td>
 </tr>
 <tr>
   <td> 12500</td><td> 3125000</td><td> 1628s</td><td> 125.9s</td><td> 12.93x</td></td><td> 18.32s</td><td> 88.86x</td>
 </tr>
</table>

<hr/>
<b>Author: Mateusz Bojanowski</b>
