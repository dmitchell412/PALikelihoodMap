
clear all;
close all

%% Input
npixel =256;

% absorption image , scattering parameter, and anisotropy
muaimage = 500* rand(npixel ,npixel ); %1/m
mus = 14000; % 1/m
anistropy = .9; % dimensionless
spacing = [.004  .004 1] ; % m 
mutr  = muaimage + mus * (1.0 - anistropy );
mueffimage = sqrt(3.0 * mutr * muaimage);
mueffimage(1:100,:) = 0.0;

% plot input
handle1 = figure(1);
imagesc(mueffimage)
colorbar

% laser location using roi mask
roimask = zeros(npixel ,npixel );
roimask(95:100,126:130) = 1;

% plot input
handle2 = figure(2);
imagesc(roimask,[0 1])
colorbar

%% Query the gpu device
% GPU must be reset on out of bounds errors
% reset(gpuDevice(1))
deviceInfo = gpuDevice(1);
numSMs = deviceInfo.MultiprocessorCount;

%% Compile and setup thread grid
% grid stride loop design pattern, 1-d grid
% http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
ssptx = parallel.gpu.CUDAKernel('sdaFluenceModel.ptx', 'sdaFluenceModel.cu');
ssptx.GridSize =[numSMs*8 1];
threadsPerBlock= 768;
ssptx.ThreadBlockSize=[threadsPerBlock  1]

% relative fluence power
Power = 1.0;

% pasignal =  fluence x mua
pasignalimage = PaSignal(ssptx,mueffimage ,spacing,Power, roimask);
handle3 = figure(3);
imagesc(pasignalimage ,[0 1])
colorbar
