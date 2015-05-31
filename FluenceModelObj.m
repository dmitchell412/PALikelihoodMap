function ObjectiveFunctionValue = FluenceModelObj(VolumeFraction,ssptx,d_pasource,muaFraction, muaReference,d_materialID,d_PAData,nsource,powerFraction,PowerFnc,d_xloc,d_yloc,d_zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz,PlotSolution)

% get power setting
power = PowerFnc(powerFraction);

%% TODO - error check same length
muaHHb     = muaFraction *muaReference*[ 3.3333,2.6667,2.6667 ,1  , 1.0667,0.66667];% [1/m]
muaHbO2    = muaFraction *muaReference*[  1    ,1.1667,1.3333 ,1.5, 1.6667,2.0];% [1/m]


%  write files as mm
ConvertToMM   = 1.e3;

%% @article{wray1988characterization,
%%   title={Characterization of the near infrared absorption spectra of
%% cytochrome aa3 and haemoglobin for the non-invasive monitoring of cerebral
%% oxygenation},
%%   author={Wray, Susan and Cope, Mark and Delpy, David T and Wyatt, John S
%% and Reynolds, E Osmund R},
%%   journal={Biochimica et Biophysica Acta (BBA)-Bioenergetics},
%%   volume={933},
%%   number={1},
%%   pages={184--192},
%%   year={1988},
%%   publisher={Elsevier}
%% }
%% 
%% 
%% @inproceedings{needles2010development,
%%   title={Development of a combined photoacoustic micro-ultrasound system for
%% estimating blood oxygenation},
%%   author={Needles, A and Heinmiller, A and Ephrat, P and Bilan-Tracey, C and
%% Trujillo, A and Theodoropoulos, C and Hirson, D and Foster, FS},
%%   booktitle={Ultrasonics Symposium (IUS), 2010 IEEE},
%%   pages={390--393},
%%   year={2010},
%%   organization={IEEE}
%% }
% objective function is l2 distance from each wavelength
ObjectiveFunctionValue = 0.0;
for idwavelength= 1:length(muaHHb)
    [d_pasource ] = feval(ssptx,d_materialID,VolumeFraction, muaHHb(idwavelength),muaHbO2(idwavelength), nsource, power ,d_xloc,d_yloc,d_zloc, d_pasource,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);
    l2distance = norm( d_pasource(:)-d_PAData(:,idwavelength) ) ;
    ObjectiveFunctionValue = ObjectiveFunctionValue + l2distance * l2distance ;
    if(PlotSolution)
       h_pasource    = gather(d_pasource );
       pasolnnii = make_nii(h_pasource,ConvertToMM *[spacingX,spacingY,spacingZ],[],[],'pasoln');
       save_nii(pasolnnii,['pasoln.' sprintf('%04d',idwavelength) '.nii.gz']) ;
       savevtkcmd = ['c3d  pasoln.' sprintf('%04d',idwavelength) '.nii.gz -o pasoln.' sprintf('%04d',idwavelength) '.vtk; sed -i ''s/scalars/pasoln/g'' pasoln.' sprintf('%04d',idwavelength) '.vtk '];
       [status result] = system(savevtkcmd);
       %% handle = figure(length(muaHHb)+ idwavelength);
       %% imagesc(h_pasource(:,:,idslice ),PAPlotRange)
       %% colorbar
    end
end
%disp([VolumeFraction,power,ObjectiveFunctionValue]);

% http://en.wikipedia.org/wiki/Multivariate_normal_distribution
% L2 norm of distance is normalized to gaussian in calling routine.
% The AcceptanceProbability = exp( (oldenergy-newenergy)/(k*T) ); 
%   is computed on the L2 distance directly to AVOID rounding errors

% datadimension = npixelx*npixely*npixelz*length(muaHHb);
% ModelProbability  = exp(-.5*ObjectiveFunctionValue )/sqrt( (2 * pi)^datadimension );
% NOTE - Gaussian normalization factors CANCEL
%     1/2 is added to normalize Gaussian distribution
%        normalize by variance to get acceptance probability
Variance = 1.e9;
ObjectiveFunctionValue = 0.5 *ObjectiveFunctionValue/Variance ; 

if(PlotSolution)
  % plot volume fraction solution
  VolumeFractionImg = double( gather(d_materialID));
  ntissue = max(max(max(VolumeFractionImg)));
  for iii = 1:ntissue
     VolumeFractionImg(VolumeFractionImg== iii  ) = VolumeFraction(iii);
  end
  volumefractionsolnnii = make_nii(VolumeFractionImg, ConvertToMM * [spacingX,spacingY,spacingZ],[],[],'volumefraction');
  save_nii(volumefractionsolnnii,'volumefractionsoln.nii.gz') ;
  savevtkcmd = ['c3d volumefractionsoln.nii.gz -o volumefractionsoln.vtk ; sed -i ''s/scalars/volfrac/g'' volumefractionsoln.vtk '];
  [status result] = system(savevtkcmd);
end
