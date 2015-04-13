%% Embarrassing Parallel GPU Greens Function Linear Super Position
clear all
close all
format shortg

%% Simulate disjoint material/tissue types
% create npixel^3 image
npixel   = 100;
materialID = int32(10*phantom3d('Modified Shepp-Logan',npixel));
materialID(materialID == 3  ) = 1;
materialID(materialID == 10 ) = 3;
handle5 = figure(5);
imagesc(materialID(:,:,npixel/2),[0 3])
colorbar

%% Query the device
% GPU must be reset on out of bounds errors
% reset(gpuDevice(1))
deviceInfo = gpuDevice(1);
numSMs = deviceInfo.MultiprocessorCount;

spacingX = 1.0e-3;
spacingY = 1.0e-3;
spacingZ = 1.0e-3;

%% mueff at each wave length
%% TODO - error check same length
mueffHHb   = [7.e02 , 4.e02];
mueffHbO2  = [5.e02 , 6.e02];
sigmamueff = 30;


%% Setup Material Parameters
ntissue = 4;
nsource    = 10;
xloc       = npixel/2*spacingX+spacingX*linspace(1,nsource ,nsource )+1.e-4;
yloc       = npixel/2*spacingY+spacingY*linspace(1,nsource ,nsource )+1.e-4;
zloc       = npixel/2*spacingZ+spacingZ*linspace(1,nsource ,nsource )+1.e-4;
power      = 10.;

%% initialize data arrays
% initialize on host and perform ONE transfer from host to device
h_pasource     = zeros(npixel,npixel,npixel);
d_pasource      = gpuArray( h_pasource  );

%% Compile and setup thread grid
% grid stride loop design pattern, 1-d grid
% http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
ssptx = parallel.gpu.CUDAKernel('sdaFluenceModel.ptx', 'sdaFluenceModel.cu');
threadsPerBlock = 256;
ssptx.ThreadBlockSize=[threadsPerBlock  1];
ssptx.GridSize=[numSMs*32               1];

%% Run on GPU
mueff      = [5.e02 , 4.e02 , 3.e02];
[d_pasource ] = feval(ssptx,ntissue,materialID, mueff, nsource, power ,xloc,yloc,zloc, d_pasource,spacingX,spacingY,spacingZ,npixel,npixel,npixel);

%%  transfer device to host
h_pasource  = gather( d_pasource  );

%%  plot pasource
handle6 = figure(6);
imagesc(h_pasource(:,:,50), [0 2.e5]);
colormap default
colorbar

%% UQ grid 
mixinggrid = [.2, .4, .6, .8]

%%  global search and plot exhaustive search
tic
statfigurenames = {'meanpasourcelo','varpasourcelo','meanpasourcehi','varpasourcehi'}
for idLambda = 1:numel(mueffHbO2) 
  %% UQ grid 
  HHbgrid  =  sigmamueff * [-2,-1,0,1,2] + mueffHHb(idLambda)
  HbO2grid =  sigmamueff * [-2,-1,0,1,2] + mueffHbO2(idLambda)
  TotalIter = numel(mixinggrid) * numel(mixinggrid) * numel(mixinggrid) * numel(HHbgrid) * numel(HbO2grid);
  IterCount = 0; 
  % mean
  d_pasourceMean  = gpuArray( zeros(npixel,npixel,npixel)  );
  for idOne = 1: numel(mixinggrid)
    % show iterations
    disp(sprintf('iter %d',idOne));
    for idTwo = 1: numel(mixinggrid)
      for idThree = 1: numel(mixinggrid)
        for idFour = 1: numel(HHbgrid)
          for idFive = 1: numel(HbO2grid)
             mueff = [   mixinggrid(idOne)   *HHbgrid( idFour)+ (1-mixinggrid(idOne))  *HbO2grid(idFive),...
                         mixinggrid(idTwo)   *HHbgrid( idFour)+ (1-mixinggrid(idTwo))  *HbO2grid(idFive),...
                         mixinggrid(idThree) *HHbgrid( idFour)+ (1-mixinggrid(idThree))*HbO2grid(idFive)];
             IterCount =  IterCount +1;  
             if mod(IterCount,100 )==0
               disp(sprintf(' %d %f %f %f ', IterCount, mueff));
             end
             [p_pasource] = feval(ssptx,ntissue,materialID, mueff, nsource, power ,xloc,yloc,zloc, d_pasource,spacingX,spacingY,spacingZ,npixel,npixel,npixel);
             d_pasourceMean  = d_pasourceMean  + 1./TotalIter * p_pasource;
  end
  end
  end
  end
  end
  IterCount = 0; 
  % variance
  d_pasourceVar  = gpuArray( zeros(npixel,npixel,npixel)  );
  for idOne = 1: numel(mixinggrid)
    % show iterations
    disp(sprintf('iter %d',idOne));
    for idTwo = 1: numel(mixinggrid)
      for idThree = 1: numel(mixinggrid)
        for idFour = 1: numel(HHbgrid)
          for idFive = 1: numel(HbO2grid)
             mueff = [   mixinggrid(idOne)   *HHbgrid( idFour)+ (1-mixinggrid(idOne))  *HbO2grid(idFive),...
                         mixinggrid(idTwo)   *HHbgrid( idFour)+ (1-mixinggrid(idTwo))  *HbO2grid(idFive),...
                         mixinggrid(idThree) *HHbgrid( idFour)+ (1-mixinggrid(idThree))*HbO2grid(idFive)];
             IterCount =  IterCount +1;  
             if mod(IterCount,100 )==0
               disp(sprintf(' %d %f %f %f ', IterCount, mueff));
             end
             [p_pasource] = feval(ssptx,ntissue,materialID, mueff, nsource, power ,xloc,yloc,zloc, d_pasource,spacingX,spacingY,spacingZ,npixel,npixel,npixel);
             d_pasourceVar= d_pasourceVar + 1./(TotalIter-1) * (p_pasource - d_pasourceMean  ).*(p_pasource - d_pasourceMean  );
 end
 end
 end
 end
 end
 % gather
 host_Mean = gather(d_pasourceMean );
 host_Std  = sqrt(gather(d_pasourceVar  )); 

 % plot
 idplot  =  2*(idLambda-1) +1;
 handleid = figure( idplot   );
 imagesc(host_Mean(:,:,50), [0 2.e+05]);
 colormap default
 colorbar
 saveas(handleid,statfigurenames{idplot },'png')

 % plot
 idplot =  2*idLambda;
 handleid = figure(  idplot );
 % gather
 imagesc(host_Std(:,:,50), [0 1.e5]);
 colormap default
 colorbar
 saveas(handleid,statfigurenames{idplot },'png')
end
toc

saveas(handle5,'material','png')
saveas(handle6,'pasource','png')
