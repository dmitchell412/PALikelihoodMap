%  input
% mua  N x N image    units - 1/m
% mask  N x N image of laser location
% relative fluence = 1.0
% mus const units -  1/m
% spacing [ dx dy ] 

%  output
% fluence N x N image

function fluence = PaSignal(ssptx,mueffimage,spacing,power, roimask)

% image dimensions
spacingX = spacing(1);
spacingY = spacing(2);
spacingZ = spacing(3);
[npixelx, npixely, npixelz] = size(mueffimage);

% identify roi locations of laser source
[rows,cols,depth] = ind2sub(size(roimask),find(roimask));
nsource    = length(rows);
d_xloc       = gpuArray(1.e-3+ spacingX* rows );
d_yloc       = gpuArray(1.e-3+ spacingY* cols );
d_zloc       = gpuArray(1.e-3+ spacingZ* depth); 
h_pasource   = zeros(size(mueffimage));
d_pasource   = gpuArray( h_pasource  );

% setup gpu input
ntissue = length(mueffimage(:));
materialID = int32([1:ntissue ]);
[d_pasource ] = feval(ssptx,ntissue,materialID, mueffimage, nsource, power ,d_xloc,d_yloc,d_zloc, d_pasource,spacingX,spacingY,spacingZ,npixelx,npixely,npixelz);

fluence = gather( d_pasource  );

