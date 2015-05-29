/*
 * Example Matlab cuda kernel interface.
 */


__device__
void pointSource(double dist, double mueff, double Power, double *paSource )
{

   double PI_Var = 3.141592653589793;
   *paSource = Power/PI_Var/4.*mueff*mueff* exp (-mueff*dist) / dist;
}
__device__
void DebugWrite(int idx,int idmat,double rad,double omega, double conduction, double mueff,double temp)
{
   printf("%d %d %12.5e %12.5e %12.5e %12.5e %12.5e\n",idx,idmat,rad,omega,conduction,mueff,temp);
   //int j,k;

   //for (j=0;j<n;j++) {
   //   for (k=0;k<n+1;k++) {
   //      printf("%d %d %12.5e %12.5e ",k,j,a[k][j].real(),a[k][j].imag());
   //   }
   //   printf(" | %d  %12.5e %12.5e \n",j,x[j].real(),x[j].imag());
   //}
   //printf("\n");
}

/*
 * Device code
 */
__global__ 
void sdaFluenceModel(
         const    int* MaterialID,
         const double* VolumeFraction,
         const double  muaHHb ,
         const double  muaHbOTwo ,
         int const NSource,
         double const Power,
         const double* SourceXloc,
         const double* SourceYloc,
         const double* SourceZloc,
               double* d_PASourceArray,
         double const  SpacingX,
         double const  SpacingY,
         double const  SpacingZ,
         int const NpixelX,
         int const NpixelY,
         int const NpixelZ)
{
    /*
      grid stride loop design pattern, 1-d grid
      http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
         - By using a loop, you can support any problem size even if it exceeds the largest grid size your CUDA device supports. Moreover, you can limit the number of blocks you use to tune performance.
    */
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < NpixelX * NpixelY * NpixelZ;
         idx += blockDim.x * gridDim.x) 
      {
        // compute indices
        int index = idx; // use dummy variable
        int kkk = index/(NpixelX*NpixelY); 
        index -= kkk*NpixelX*NpixelY; 
        
        int jjj = index/NpixelX; 
        index -= jjj*NpixelX; 
        
        int iii = index/1;

        /* get material parameters */
        int const idmaterial =  MaterialID[idx];
        double Anisotropy =.9;
        double OpticalScattering =  14000.0; // [1/m]
        double OpticalAbsorption = muaHHb *         VolumeFraction[idmaterial] 
                                 + muaHbOTwo * (1.0-VolumeFraction[idmaterial]);
        double mutr   = OpticalAbsorption  + OpticalScattering * (1.0 - Anisotropy);
        double mueff  = sqrt(3.0 * mutr * OpticalAbsorption );

        // linear superpostion of fluence sources
        double fluence = 0.0;
        for (int lll=0;lll<NSource;lll++) 
          {
           double radiusSQ = (iii * SpacingX - SourceXloc[lll])*(iii * SpacingX - SourceXloc[lll])
                           + (jjj * SpacingY - SourceYloc[lll])*(jjj * SpacingY - SourceYloc[lll])
                           + (kkk * SpacingZ - SourceZloc[lll])*(kkk * SpacingZ - SourceZloc[lll]);
           double radius   = sqrt(radiusSQ);
           // call GF code 
           double sourcefluence;
           pointSource(radius, mueff,  Power , &sourcefluence);
           // DebugWrite(idx,idmaterial,radius,omega,conduction,mueff,sourcefluence);
           // superposition
           fluence = fluence + sourcefluence/((double)NSource); 
          }
        // store fluence in array
        d_PASourceArray[idx] = fluence;
      }
}


