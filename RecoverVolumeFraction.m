clear all
close all
format shortg

%% Spectral inversion for multiple wavelengths
WaveLength = [680   , 710   , 750   , 850   , 920  , 950  ];
NWavelength = length(WaveLength);

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
handle1 = figure(2*NWavelength+1);
imagesc(materialID(:,:,idslice ),[0 5])
colorbar

%% load tumor mask
tumormask =load_untouch_nii('tumormask.nii.gz');
maskimage = double(tumormask.img);

PAPlotRange = [0 300];
% load data
disp('loading PA data');
h_PAData = zeros(npixelx* npixely* npixelz,NWavelength);  
for idwavelength = 1:NWavelength
  padatanii = load_untouch_nii(['padata.' sprintf('%04d',idwavelength) '.nii.gz']) ;
  % view data
  handle = figure(idwavelength);
  imagesc(maskimage(:,:,idslice).*padatanii.img(:,:,idslice ),PAPlotRange )
  colorbar
  % store data array
  h_PAData(:,idwavelength) = maskimage(:).*padatanii.img(:) ;
end

%% Get laser source locations
lasersource  = load_untouch_nii('lasersource.nii.gz');
[rows,cols,depth] = ind2sub(size(lasersource.img),find(lasersource.img));
nsource    = length(rows);
PowerLB    = .0001;
PowerUB    =  .001;
PowerFnc   = @(x) PowerLB + x * (PowerUB-PowerLB);

%% Query the device
% GPU must be reset on out of bounds errors
% reset(gpuDevice(1))
deviceInfo = gpuDevice(1);
numSMs = deviceInfo.MultiprocessorCount;

%% initialize data arrays
% initialize on host and perform ONE transfer from host to device
tic;
h_pasource   = zeros(npixelx,npixely,npixelz);
d_pasource   = gpuArray( h_pasource  );
d_PAData     = gpuArray( h_PAData    );
d_materialID = gpuArray( materialID  );
d_xloc       = gpuArray(1.e-3+ spacingX* rows );
d_yloc       = gpuArray(1.e-3+ spacingY* cols );
d_zloc       = gpuArray(1.e-3+ spacingZ* depth); 
transfertime = toc;
disp(sprintf('transfer time to device %f',transfertime));

%% Compile and setup thread grid
% grid stride loop design pattern, 1-d grid
% http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
ssptx = parallel.gpu.CUDAKernel('sdaSpectralFluenceModel.ptx', 'sdaSpectralFluenceModel.cu');
ssptx.GridSize =[numSMs*8 1];
threadsPerBlock= 768;
ssptx.ThreadBlockSize=[threadsPerBlock  1]

%% create anonymous function
muaReference     = 3.e1; % [1/m]
% TODO - change function signature to use struct
loss = @(x) FluenceModelObj([0,x(1:length(x)-2)],ssptx,d_pasource,x(length(x)),muaReference,d_materialID,d_PAData,nsource,x(length(x)-1),PowerFnc,d_xloc,d_yloc,d_zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz,0);

%% % tune kernel
%% for blockpergrid = [numSMs*8,numSMs*16,numSMs*32,numSMs*48,numSMs*64];
%% for threadsPerBlock = [128,256,512,768];
%%   ssptx.GridSize=[blockpergrid 1];
%%   ssptx.ThreadBlockSize=[threadsPerBlock  1];
%%   tic;
%%   f = loss(InitialGuess)
%%   kernelruntime = toc;
%%   disp(sprintf('blockpergrid=%d  threadsPerBlock=%d runtime=%f',blockpergrid,threadsPerBlock,kernelruntime));
%% end
%% end

%% run opt solver
disp('starting solver')
options = anneal();
%options.MaxTries = 2;
%options.MaxConsRej = 1;

% use least square direction for proposal distribution
% TODO - debug
% TODO - change function signature to use struct
options.Generator =  @(x) StochasticNewton([0,x(1:length(x)-1)],ssptx,d_pasource,muaHHb, muaHbO2,d_materialID,d_PAData,nsource,x(length(x)),PowerRange,d_xloc,d_yloc,d_zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);

% TODO - use Monte Carlo for now
options.Generator = @(x) rand(1,length(x));

% set plotting function
options.PlotLoss =  @(x) FluenceModelObj([0,x(1:length(x)-2)],ssptx,d_pasource,x(length(x)),muaReference,d_materialID,d_PAData,nsource,x(length(x)-1),PowerFnc,d_xloc,d_yloc,d_zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz,1);

% uniformly search parameter space to find good initial guess
RandomInitialGuess = 10;
RandomInitialGuess = 1;
tic;
for iii = 1:RandomInitialGuess  % embarrasingly parallel on initial guess

  %% initial guess
  %% materialID[0] not used
  %% assume 50/50 volume fraction initially
  %% last entry is percent power
  InitialGuess = [.5*ones(1,ntissue),.6,.9];

  [SolnVector FunctionValue ] = anneal(loss,InitialGuess,options);
  % TODO store best solution
end
mcmcruntime = toc;
disp(sprintf('mcmc run time %f',mcmcruntime) );

%SolnVector = [ 7.2e-01 9.9e-01 1.9e-01 5.9e-01 7.0e-03 8.9e-02 2.9e-02];
%f = plotloss(SolnVector )


