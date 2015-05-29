clear all
close all
format shortg

%
disp('loading GMM tissue types');
tumorlabel  = load_untouch_nii('tumor_ATROPOS_GMM.nii.gz');
materialID = int32(tumorlabel.img);
%materialID(materialID == 0  ) = 1;

%% Initial Guess for volume fractions
ntissue = max(max(max(materialID)));

[npixelx, npixely, npixelz] = size(materialID);
spacingX = tumorlabel.hdr.dime.pixdim(2)*1.e-3;
spacingY = tumorlabel.hdr.dime.pixdim(3)*1.e-3;
spacingZ = tumorlabel.hdr.dime.pixdim(4)*1.e-3;

idslice = 11;
handle5 = figure(5);
imagesc(materialID(:,:,idslice ),[0 5])
colorbar

%% mua for each species at each wave length  
%% TODO - error check same length
muaHHb   = [7.e02 ,7.e02 ,7.e02 ,7.e02 ,7.e02 , 4.e02]; % [1/m]
muaHbO2  = [5.e02 ,5.e02 ,5.e02 ,5.e02 ,5.e02 , 6.e02]; % [1/m]

%% load tumor mask
tumormask =load_untouch_nii('tumormask.nii.gz');

% load data
disp('loading PA data');
PAData = zeros(npixelx* npixely* npixelz,muaHHb);  
for iii = 1: length(muaHHb)
  padatanii = load_untouch_nii(['padata.' sprintf('%04d',iii) '.nii.gz']) ;
  PAData(:,iii) = tumormask.img(:).*padatanii.img(:) ;
end

%% Setup laser source locations
lasersource  = load_untouch_nii('lasersource.nii.gz');
[rows,cols,depth] = ind2sub(size(lasersource.img),find(lasersource.img));
nsource    = length(rows);
xloc       = spacingX* rows;
yloc       = spacingY* cols;
zloc       = spacingZ* depth; 
power      = 100.;

%% Query the device
% GPU must be reset on out of bounds errors
% reset(gpuDevice(1))
deviceInfo = gpuDevice(1);
numSMs = deviceInfo.MultiprocessorCount;

%% Compile and setup thread grid
% grid stride loop design pattern, 1-d grid
% http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
ssptx = parallel.gpu.CUDAKernel('sdaSpectralFluenceModel.ptx', 'sdaSpectralFluenceModel.cu');

% tune kernel
for gridsize = [numSMs*16,numSMs*32,numSMs*48,numSMs*64];
for threadsPerBlock = [256,512,768];
  ssptx.GridSize=[gridsize 1];
  ssptx.ThreadBlockSize=[threadsPerBlock  1]
  tic;
  VolumeFraction = rand(ntissue ,1);
  f = FluenceModelObj(VolumeFraction,ssptx,muaHHb, muaHbO2,materialID,PAData,nsource,power,xloc,yloc,zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz)
  toc;
end
end

