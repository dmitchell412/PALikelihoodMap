function ObjectiveFunctionValue = FluenceModelObj(VolumeFraction,ssptx,muaHHb, muaHbOTwo,materialID,PAData,nsource,power,xloc,yloc,zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz)

%% initialize data arrays
% initialize on host and perform ONE transfer from host to device
h_pasource     = zeros(npixelx,npixely,npixelz);
d_pasource      = gpuArray( h_pasource  );

% objective function is l2 distance from each wavelength
ObjectiveFunctionValue = 0.0;
for iii= 1:length(muaHHb)
    [d_pasource ] = feval(ssptx,materialID,VolumeFraction, muaHHb(iii),muaHbOTwo(iii), nsource, power ,xloc,yloc,zloc, d_pasource,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);
    h_pasource = gather(d_pasource);
    ObjectiveFunctionValue = ObjectiveFunctionValue + norm( h_pasource(:)-PAData(:,iii) ) ;
end
