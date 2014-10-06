/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */
// #include <cusp/complex.h>
// 
// // borrow petsc types
// typedef double PetscScalar;
// typedef cusp::complex<PetscScalar> PetscComplex;
// typedef int PetscInt;

// FIXME global var used to pass array of data
int const MaxSpecies = 2;
int const MaxSize    = 2*MaxSpecies ;
int const tol        = 1.E-4;
int const upperbound = 100;


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
         const double* Perfusion,
         const double* ThermalConduction,
         const double* EffectiveAttenuation,
         const    int* MaterialID,
         const double* SourceXloc,
         const double* SourceYloc,
         const double* SourceZloc,
               double* d_TemperatureArray,
         double const Power,
         double const AterialTemperature,
         int const NSource,
         int const NTissue,
         int const Npixel)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;
    int const idmaterial =  MaterialID[idx];

    double omega      = SpecificHeatBlood * Perfusion[idmaterial];
    double conduction = ThermalConduction[idmaterial];
    double mueff      = EffectiveAttenuation[idmaterial];

    // linear superpostion of temperature sources
    double temperature = 0.0;
    for (int iii=0;iii<NSource;iii++) 
      {
       s1 = 3.0/4.0/PI*P*mua*mutr/(w-k*mueff*mueff)*exp(-mueff*r)/r+ua;      
       s2 = s1;      
       s5 = 1/r*exp(sqrt(w/k)*r)*(-4.0*sqrt(w/k)*R2*exp(-sqrt(w/k)*R2)*u0*PI*R1*w+4.0*sqrt(w/k)*R2*exp(-sqrt(w/k)*R2)*u0*PI*R1*k*mueff*mueff+3.0*sqrt(w/k)*R2*P*mua*mutr*exp(-sqrt(w/k)*R2-mueff*R1)+4.0*sqrt(w/k)*R2*exp(-sqrt(w/k)*R2)*ua*PI*R1*w-4.0*sqrt(w/k)*R2*exp(-sqrt(w/k)*R2)*ua*PI*R1*k*mueff*mueff-3.0*P*mua*mutr*mueff*R2*exp(-mueff*R2-sqrt(w/k)*R1)-3.0*P*mua*mutr*exp(-mueff*R2-sqrt(w/k)*R1)+4.0*exp(-sqrt(w/k)*R2)*ua*PI*R1*w-4.0*exp(-sqrt(w/k)*R2)*u0*PI*R1*w+4.0*exp(-sqrt(w/k)*R2)*u0*PI*R1*k*mueff*mueff+3.0*P*mua*mutr*exp(-sqrt(w/k)*R2-mueff*R1)-4.0*exp(-sqrt(w/k)*R2)*ua*PI*R1*k*mueff*mueff)/4.0;      
       s6 = exp(-sqrt(w/k)*(-R1+R2))/(-w+k*mueff*mueff)/PI/(exp(-2.0*sqrt(w/k)*(-R1+R2))+sqrt(w/k)*R2*exp(-2.0*sqrt(w/k)*(-R1+R2))-1.0+sqrt(w/k)*R2);      
       s4 = s5*s6;      
       s6 = 1/r*exp(-sqrt(w/k)*r)*exp(-sqrt(w/k)*(-R1+R2))/4.0;      
       s9 = 4.0*exp(sqrt(w/k)*R2)*u0*PI*R1*w-4.0*exp(sqrt(w/k)*R2)*u0*PI*R1*w*sqrt(w/k)*R2-4.0*exp(sqrt(w/k)*R2)*u0*PI*R1*k*mueff*mueff+4.0*exp(sqrt(w/k)*R2)*u0*PI*R1*k*mueff*mueff*sqrt(w/k)*R2-3.0*P*mua*mutr*exp(sqrt(w/k)*R2-mueff*R1)+3.0*P*mua*mutr*sqrt(w/k)*R2*exp(sqrt(w/k)*R2-mueff*R1)-4.0*exp(sqrt(w/k)*R2)*ua*PI*R1*w+4.0*exp(sqrt(w/k)*R2)*ua*PI*R1*w*sqrt(w/k)*R2+4.0*exp(sqrt(w/k)*R2)*ua*PI*R1*k*mueff*mueff-4.0*exp(sqrt(w/k)*R2)*ua*PI*R1*k*mueff*mueff*sqrt(w/k)*R2+3.0*P*mua*mutr*mueff*R2*exp(sqrt(w/k)*R1-mueff*R2)+3.0*P*mua*mutr*exp(sqrt(w/k)*R1-mueff*R2);      
       s10 = 1/(exp(-2.0*sqrt(w/k)*(-R1+R2))+sqrt(w/k)*R2*exp(-2.0*sqrt(w/k)*(-R1+R2))-1.0+sqrt(w/k)*R2);
       temperature = temperature + s10; 
      }
    d_TemperatureArray[idx] = temperature;

}


