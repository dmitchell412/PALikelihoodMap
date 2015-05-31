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
PowerLB    = .5;
PowerUB    =  2.0;
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
loss = @(x) FluenceModelObj([0,x(1:length(x)-2)],ssptx,d_pasource,x(length(x)),muaReference,d_materialID,d_PAData,nsource,x(length(x)-1),PowerFnc,d_xloc,d_yloc,d_zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);

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
options.Generator =  @(x) rand(1,length(x));

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

  [SolnVector FunctionValue opthistory] = anneal(loss,InitialGuess,options);
  % TODO store best solution
end
mcmcruntime = toc;
disp(sprintf('mcmc run time %f',mcmcruntime) );

% write optimization history for scatter plot in R
csvwrite('opthistory.csv',opthistory);

%SolnVector = [ 6.758e-02,4.987e-01,3.796e-01,1.657e-01,8.022e-01,1.084e-03 ];
%SolnVector = [ 0.95875  ,  0.77622,0.10041  , 0.067474,0.25271  ,0.0019434 ]; 
%SolnVector = [ 1.48e-01 , 2.95e-02, 1.60e-01, 2.63e-01, 6.43e-01, 1.57e-03 ];
%SolnVector = [ 0.48858  , 0.25659 , 0.37129 , 0.16478 , 0.02466 , 0.38788  ];
%f = loss(SolnVector )

% plot volume fraction solution
VolumeFraction = [0,SolnVector(1:length(SolnVector)-2)]; 
VolumeFractionImg = double( materialID);
for iii = 1:ntissue
   VolumeFractionImg(materialID == iii  ) = VolumeFraction(iii);
end
volumefractionsolnnii = make_nii(VolumeFractionImg,tumorlabel.hdr.dime.pixdim(2:4),[],[],'volumefraction');
save_nii(volumefractionsolnnii,'volumefractionsoln.nii.gz') ;
savevtkcmd = ['c3d volumefractionsoln.nii.gz -o volumefractionsoln.vtk ; sed -i ''s/scalars/volfrac/g'' volumefractionsoln.vtk '];
[status result] = system(savevtkcmd);

% plot predict PA signal
power      = PowerFnc(SolnVector(  length(SolnVector) -1 ));
muaHHb     = SolnVector(  length(SolnVector) )    *muaReference*[ 3.3333,2.6667,2.6667 ,1  , 1.0667,0.66667];% [1/m]
muaHbO2    = SolnVector(  length(SolnVector) )    *muaReference*[  1    ,1.1667,1.3333 ,1.5, 1.6667,2.0];% [1/m]

for idwavelength= 1:NWavelength
  [d_pasource ] = feval(ssptx,d_materialID,VolumeFraction, muaHHb(idwavelength),muaHbO2(idwavelength), nsource, power ,d_xloc,d_yloc,d_zloc, d_pasource,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);
  h_pasource    = gather(d_pasource );
  pasolnnii = make_nii(h_pasource,tumorlabel.hdr.dime.pixdim(2:4),[],[],'pasoln');
  save_nii(pasolnnii,['pasoln.' sprintf('%04d',idwavelength) '.nii.gz']) ;
  savevtkcmd = ['c3d  pasoln.' sprintf('%04d',idwavelength) '.nii.gz -o pasoln.' sprintf('%04d',idwavelength) '.vtk; sed -i ''s/scalars/pasoln/g'' pasoln.' sprintf('%04d',idwavelength) '.vtk '];
  [status result] = system(savevtkcmd);
  handle = figure(NWavelength+ idwavelength);
  imagesc(h_pasource(:,:,idslice ),PAPlotRange)
  colorbar
end

