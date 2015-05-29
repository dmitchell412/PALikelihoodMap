function ObjectiveFunctionValue = FluenceModelObj(VolumeFraction,ssptx,d_pasource,muaHHb, muaHbO2,d_materialID,d_PAData,nsource,power,d_xloc,d_yloc,d_zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz)

% objective function is l2 distance from each wavelength
ObjectiveFunctionValue = 0.0;
for idwavelength= 1:length(muaHHb)
    [d_pasource ] = feval(ssptx,d_materialID,VolumeFraction, muaHHb(idwavelength),muaHbO2(idwavelength), nsource, power ,d_xloc,d_yloc,d_zloc, d_pasource,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);
    ObjectiveFunctionValue = ObjectiveFunctionValue + norm( d_pasource(:)-d_PAData(:,idwavelength) ) ;
end
