clear all
close all
format shortg

% create npixel^3 image
npixel   = 100;
materialID = int32(10*phantom3d('Modified Shepp-Logan',npixel));
materialID(materialID == 3  ) = 1;
materialID(materialID == 10 ) = 3;
handle1 = figure(1)
imagesc(materialID(:,:,npixel/2),[0 3])
colorbar

% query the device
deviceInfo = gpuDevice(1);
numSMs = deviceInfo.MultiprocessorCount;
% % GPU must be reset on out of bounds errors
% reset(gpuDevice(1))

spacingX = 1.0e-3;
spacingY = 1.0e-3;
spacingZ = 1.0e-3;

%%diary
ntissue = 4;
perfusion  = [5.e01 , 4.e01 , 3.e01, 6.e01];
conduction = [5.e-1 , 4.e-1 , 3.e-1, 6.e-1];
mueff      = [5.e02 , 4.e02 , 3.e02, 6.e02];
nsource    = 10;
xloc       = npixel/2*spacingX+spacingX*linspace(1,nsource ,nsource )+1.e-4;
yloc       = npixel/2*spacingY+spacingY*linspace(1,nsource ,nsource )+1.e-4;
zloc       = npixel/2*spacingZ+spacingZ*linspace(1,nsource ,nsource )+1.e-4;
u_artery   = 37.;
c_blood    = 3480.;
power      = 10.;

% initialize data on host
h_temperature     = zeros(npixel,npixel,npixel);
% perform ONE transfer from host to device
d_temperature  = gpuArray( h_temperature  );

R1 = .001 ; % 1mm
R2 = .1   ; % 100mm
%%  compile and run
ssptx = parallel.gpu.CUDAKernel('steadyStatePennesLaser.ptx', 'steadyStatePennesLaser.cu');
threadsPerBlock = 256;
ssptx.ThreadBlockSize=[threadsPerBlock  1];
ssptx.GridSize=[numSMs*32               1];
[d_temperature ] = feval(ssptx,ntissue,materialID,perfusion,conduction, mueff, R1, R2, nsource, power ,xloc,yloc,zloc, u_artery ,u_artery , c_blood, d_temperature,spacingX,spacingY,spacingZ,npixel,npixel,npixel);


%%  transfer device to host
h_temperature  = gather( d_temperature  );

%%  plot
handle2 = figure(2)
imagesc(h_temperature(:,:,50), [37 100]);
colormap default
colorbar


tic
sizesearch = 500;
objective =zeros(sizesearch,1);
for iii = 1:sizesearch
 disp(sprintf('iter %d',iii));
 mueff(2) = 1. *iii;
 [p_temperature ] = feval(ssptx,ntissue,materialID,perfusion,conduction, mueff, R1, R2, nsource, power ,xloc,yloc,zloc, u_artery ,u_artery , c_blood, d_temperature,spacingX,spacingY,spacingZ,npixel,npixel,npixel);
 objective(iii) = gather(sum((p_temperature(:) - d_temperature(:)).^2));
end
toc
handle3 = figure(3)
plot(objective)

saveas(handle1,'material','png')
saveas(handle2,'temperature','png')
saveas(handle3,'exhaustivesearch','png')
