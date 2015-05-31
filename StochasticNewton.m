function ProposalStep = StochasticNewton(VolumeFraction,ssptx,d_pasource,muaHHb, muaHbO2,d_materialID,d_PAData,nsource,power,d_xloc,d_yloc,d_zloc,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz)

% stochastic newton proposal distribution
% Martin, James, et al. "A stochastic Newton MCMC method for large-scale statistical inverse problems with application to seismic inversion." SIAM Journal on Scientific Computing 34.3 (2012): A1460-A1487.

% objective function is l2 distance from each wavelength
for idwavelength= 1:length(muaHHb)
    [d_pasource,d_jacobian ] = feval(ssptx,d_materialID,VolumeFraction, muaHHb(idwavelength),muaHbO2(idwavelength), nsource, power ,d_xloc,d_yloc,d_zloc, d_pasource,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);
    d_residual = norm( d_pasource(:)-d_PAData(:,idwavelength) ) ;
    % TODO - use QR decomposition, normal equations for now...
    QuasiNewtonDirection = -(d_jacobian'*d_jacobian) \(d_jacobian'*d_residual);
    HessianApprox = d_jacobian'*d_jacobian
end

ProposalStep = HessianApprox * randn(length(VolumeFraction)+1,1) +  [VolumeFraction,power] + QuasiNewtonDirection ;
