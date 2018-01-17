/* Simple CUDA library for APSP problem
 *
 * Author: Matuesz Bojanowski
 *  Email: bojanowski.mateusz@gmail.com
 */

#ifndef _CUDA_APSP_
#define _CUDA_APSP_

#include "../apsp.h"

/* */
int cudaNaiveFW(const std::unique_ptr<graphAPSPTopology>& data);

/* */
int cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& data);


#endif /* _APSP_ */
