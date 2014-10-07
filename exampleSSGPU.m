clear all
close all
format shortg

% create npixel^3 image
npixel   = 256;
materialID = 10.*phantom3d('Modified Shepp-Logan',npixel);
imshow(materialID(:,:,npixel/2))

% query the device
deviceInfo = gpuDevice(1);
numSMs = deviceInfo.MultiprocessorCount;
% % GPU must be reset on out of bounds errors
% reset(gpuDevice(1))

%%diary
ntissue = 4;
perfusion  = [5.e01 , 4.e01 , 3.e01, 6.e01];
conduction = [5.e-1 , 4.e-1 , 3.e-1, 6.e-1];
mueff      = [5.e02 , 4.e02 , 3.e02, 6.e02];
nsource    = 10;
xloc       = linspace(1,nsource ,nsource );
yloc       = linspace(1,nsource ,nsource );
zloc       = linspace(1,nsource ,nsource );
u_artery   = 37.;
c_blood    = 3480.;
power      = 10.;

% initialize data on host
h_temperature     = zeros(npixel,npixel,npixel);
spacingX = 1.0;
spacingY = 1.0;
spacingZ = 1.0;
% perform ONE transfer from host to device
d_temperature  = gpuArray( h_temperature  );

%%  compile and run
ssptx = parallel.gpu.CUDAKernel('steadyStatePennesLaser.ptx', 'steadyStatePennesLaser.cu');
threadsPerBlock = 256;
ssptx.ThreadBlockSize=[threadsPerBlock  1];
ssptx.GridSize=[numSMs*32               1];
[d_temperature ] = feval(ssptx,ntissue,materialID,perfusion,conduction, mueff, nsource, power ,xloc,yloc,zloc, u_artery , c_blood, d_temperature,spacingX,spacingY,spacingZ,npixel,npixel,npixel);


%%  transfer device to host
h_temperature  = gather( d_temperature  );

%%  plot
%imagesc(h_temperature(:,:,100));
