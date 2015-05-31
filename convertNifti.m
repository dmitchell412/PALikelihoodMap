clear all
close all

% load example data
exampledata = load('Processed_Data.mat')

% get spacing parameters
% FIXME - hack - spacing error ? 
PASpacing = [ exampledata.DepthPA(2) -   exampledata.DepthPA(1),...
              exampledata.WidthPA(2) -   exampledata.WidthPA(1),...
              (exampledata.ElevPA(2)  -   exampledata.ElevPA(1)) ]

BMSpacing = [ exampledata.DepthBM(2) -   exampledata.DepthBM(1),...
              exampledata.WidthBM(2) -   exampledata.WidthBM(1),...
              exampledata.ElevBM(2)  -   exampledata.ElevBM(1) ]

PAExtent = [size(exampledata.PA,1) * PASpacing(1), size(exampledata.PA,2) * PASpacing(2), size(exampledata.PA,3) * PASpacing(3)]
BMExtent = [size(exampledata.BM,1) * BMSpacing(1), size(exampledata.BM,2) * BMSpacing(2), size(exampledata.BM,3) * BMSpacing(3)]

offset = 133 * PASpacing(1) 
%save bmode data
bmodenii = make_nii(exampledata.BM,BMSpacing,[],[],'bmodedata');
save_nii(bmodenii,'bmode.nii.gz') ;
savevtkcmd  = 'c3d  bmode.nii.gz -o bmode.vtk';
system(savevtkcmd);

%save pa data at each wavelength
backgroundshift = [ 66.77914    , 55.57656    , 53.79084    , 70.75279    , 80.28145    , 75.00383   ];


for iii = 1:size(exampledata.Wave,2)
  padatanii = make_nii(exampledata.PA(:,:,:,iii) - backgroundshift(iii) ,PASpacing);
  save_nii(padatanii,['padata.' sprintf('%04d',iii) '.nii.gz']) ;
  savevtkcmd = ['c3d  padata.' sprintf('%04d',iii) '.nii.gz tumormask.nii.gz -multiply -o padata.' sprintf('%04d',iii) '.vtk; sed -i ''s/scalars/padata/g'' padata.' sprintf('%04d',iii) '.vtk']
  system(savevtkcmd);
end

resamplecmd = '/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/WarpImageMultiTransform 3 bmode.nii.gz bmoderesample.nii.gz offset.txt -R padata.0001.nii.gz';
savevtkcmd  = 'c3d  bmoderesample.nii.gz -o bmoderesample.vtk; sed -i ''s/scalars/bmode/g'' bmoderesample.vtk';
maskcmd     = '/opt/apps/itksnap/c3d-1.0.0-Linux-x86_64/bin/c3d -verbose  bmoderesample.nii.gz -thresh 50 inf 1 0 -connected-components -threshold 1 1 1 0 -dilate 1 2x2x2vox -erode 1 2x2x2vox  -o bmodemask.nii.gz';
segmentationcmd = '/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/Atropos -d 3 -a bmoderesample.nii.gz  -x bmodemask.nii.gz -i kmeans[3]  -c [3,0.0] -m [0.1,1x1x1] -k Gaussian -o [bmode_ATROPOS_GMM.nii.gz,bmode_ATROPOS_GMM_POSTERIORS%d.nii.gz]';
tumormaskcmd   = '/opt/apps/itksnap/c3d-1.0.0-Linux-x86_64/bin/c3d -verbose  bmode_ATROPOS_GMM.nii.gz -thresh 1 1 1 0 -connected-components -threshold 1 1 1 0  -dilate 1 5x5x5vox -o tumormask.nii.gz'
tumorsegmentationcmd = '/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/Atropos -d 3 -a bmoderesample.nii.gz  -x tumormask.nii.gz -i kmeans[5]  -c [3,0.0] -m [0.1,1x1x1] -k Gaussian -o [tumor_ATROPOS_GMM.nii.gz,tumor_ATROPOS_GMM_POSTERIORS%d.nii.gz]';
disp(resamplecmd ); disp(savevtkcmd  ); disp(maskcmd     ); disp(segmentationcmd);
