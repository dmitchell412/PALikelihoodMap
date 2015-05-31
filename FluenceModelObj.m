function ObjectiveFunctionValue = FluenceModelObj(VolumeFraction,ssptx,d_pasource,muaHHb, muaHbO2,d_materialID,d_PAData,nsource,power,d_xloc,d_yloc,d_zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz)

% objective function is l2 distance from each wavelength
ObjectiveFunctionValue = 0.0;
for idwavelength= 1:length(muaHHb)
    [d_pasource ] = feval(ssptx,d_materialID,VolumeFraction, muaHHb(idwavelength),muaHbO2(idwavelength), nsource, power ,d_xloc,d_yloc,d_zloc, d_pasource,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);
    l2distance = norm( d_pasource(:)-d_PAData(:,idwavelength) ) ;
    ObjectiveFunctionValue = ObjectiveFunctionValue + l2distance * l2distance ;
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
