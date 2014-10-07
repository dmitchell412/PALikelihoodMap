/*
 * Example Matlab cuda kernel interface.
 */


// __device__
// void WriteSolution(PetscComplex a[][MaxSize],int n,PetscComplex *x)
// {
//    int j,k;
// 
//    for (j=0;j<n;j++) {
//       for (k=0;k<n+1;k++) {
//          printf("%d %d %12.5e %12.5e ",k,j,a[k][j].real(),a[k][j].imag());
//       }
//       printf(" | %d  %12.5e %12.5e \n",j,x[j].real(),x[j].imag());
//    }
//    printf("\n");
// }

/*
 * Device code
 */
__global__ 
void steadyStatePennesLaser(
         int const NTissue,
         const    int* MaterialID, // FIXME - is this data type converted correctly ? 
         const double* Perfusion,
         const double* ThermalConduction,
         const double* EffectiveAttenuation,
         int const NSource,
         double const Power,
         const double* SourceXloc,
         const double* SourceYloc,
         const double* SourceZloc,
         double const AterialTemperature,
         double const SpecificHeatBlood ,
               double* d_TemperatureArray,
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
        double omega      = SpecificHeatBlood * Perfusion[idmaterial];
        double conduction = ThermalConduction[idmaterial];
        double mueff      = EffectiveAttenuation[idmaterial];

        // linear superpostion of temperature sources
        double temperature = 0.0;
        for (int lll=0;lll<NSource;lll++) 
          {
           double radiusSQ = (iii * SpacingX - SourceXloc[lll])*(iii * SpacingX - SourceXloc[lll])
                           + (jjj * SpacingY - SourceYloc[lll])*(jjj * SpacingY - SourceYloc[lll])
                           + (kkk * SpacingZ - SourceZloc[lll])*(kkk * SpacingZ - SourceZloc[lll]);
           double radius   = sqrt(radiusSQ);
           // FIXME - add GF code here
           //s1 = 3.0/4.0/PI*P*mua*mutr/(w-k*mueff*mueff)*exp(-mueff*r)/r+ua;      
           //s2 = s1;      
           //s5 = 1/r*exp(sqrt(w/k)*r)*(-4.0*sqrt(w/k)*R2*exp(-sqrt(w/k)*R2)*u0*PI*R1*w+4.0*sqrt(w/k)*R2*exp(-sqrt(w/k)*R2)*u0*PI*R1*k*mueff*mueff+3.0*sqrt(w/k)*R2*P*mua*mutr*exp(-sqrt(w/k)*R2-mueff*R1)+4.0*sqrt(w/k)*R2*exp(-sqrt(w/k)*R2)*ua*PI*R1*w-4.0*sqrt(w/k)*R2*exp(-sqrt(w/k)*R2)*ua*PI*R1*k*mueff*mueff-3.0*P*mua*mutr*mueff*R2*exp(-mueff*R2-sqrt(w/k)*R1)-3.0*P*mua*mutr*exp(-mueff*R2-sqrt(w/k)*R1)+4.0*exp(-sqrt(w/k)*R2)*ua*PI*R1*w-4.0*exp(-sqrt(w/k)*R2)*u0*PI*R1*w+4.0*exp(-sqrt(w/k)*R2)*u0*PI*R1*k*mueff*mueff+3.0*P*mua*mutr*exp(-sqrt(w/k)*R2-mueff*R1)-4.0*exp(-sqrt(w/k)*R2)*ua*PI*R1*k*mueff*mueff)/4.0;      
           //s6 = exp(-sqrt(w/k)*(-R1+R2))/(-w+k*mueff*mueff)/PI/(exp(-2.0*sqrt(w/k)*(-R1+R2))+sqrt(w/k)*R2*exp(-2.0*sqrt(w/k)*(-R1+R2))-1.0+sqrt(w/k)*R2);      
           //s4 = s5*s6;      
           //s6 = 1/r*exp(-sqrt(w/k)*r)*exp(-sqrt(w/k)*(-R1+R2))/4.0;      
           //s9 = 4.0*exp(sqrt(w/k)*R2)*u0*PI*R1*w-4.0*exp(sqrt(w/k)*R2)*u0*PI*R1*w*sqrt(w/k)*R2-4.0*exp(sqrt(w/k)*R2)*u0*PI*R1*k*mueff*mueff+4.0*exp(sqrt(w/k)*R2)*u0*PI*R1*k*mueff*mueff*sqrt(w/k)*R2-3.0*P*mua*mutr*exp(sqrt(w/k)*R2-mueff*R1)+3.0*P*mua*mutr*sqrt(w/k)*R2*exp(sqrt(w/k)*R2-mueff*R1)-4.0*exp(sqrt(w/k)*R2)*ua*PI*R1*w+4.0*exp(sqrt(w/k)*R2)*ua*PI*R1*w*sqrt(w/k)*R2+4.0*exp(sqrt(w/k)*R2)*ua*PI*R1*k*mueff*mueff-4.0*exp(sqrt(w/k)*R2)*ua*PI*R1*k*mueff*mueff*sqrt(w/k)*R2+3.0*P*mua*mutr*mueff*R2*exp(sqrt(w/k)*R1-mueff*R2)+3.0*P*mua*mutr*exp(sqrt(w/k)*R1-mueff*R2);      
           //s10 = 1/(exp(-2.0*sqrt(w/k)*(-R1+R2))+sqrt(w/k)*R2*exp(-2.0*sqrt(w/k)*(-R1+R2))-1.0+sqrt(w/k)*R2);
           double s10 = 1.0;
           temperature = temperature + s10; 
          }
        // store temperature in array
        d_TemperatureArray[idx] = temperature;
      }
}


