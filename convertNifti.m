clear all
close all

% load example data
exampledata = load('Processed_Data.mat')

% get spacing parameters
PASpacing = [ exampledata.DepthPA(2) -   exampledata.DepthPA(1),...
              exampledata.WidthPA(2) -   exampledata.WidthPA(1),...
              exampledata.ElevPA(2)  -   exampledata.ElevPA(1) ]

BMSpacing = [ exampledata.DepthBM(2) -   exampledata.DepthBM(1),...
              exampledata.WidthBM(2) -   exampledata.WidthBM(1),...
              exampledata.ElevBM(2)  -   exampledata.ElevBM(1) ]


%save bmode data
bmodenii = make_nii(exampledata.BM,BMSpacing)
save_nii(bmodenii,'bmode.nii.gz') 
savevtkcmd = 'c3d  bmode.nii.gz -o bmode.vtk'

%save pa data at each wavelength
for iii = 1:size(exampledata.Wave,2)
  padatanii = make_nii(exampledata.PA(:,:,:,iii),PASpacing)
  save_nii(padatanii,['padata.' sprintf('%04d',iii) '.nii.gz']) 
  savevtkcmd = ['c3d  padata.' sprintf('%04d',iii) '.nii.gz -o padata.' sprintf('%04d',iii) '.vtk']
  system(savevtkcmd);
end

resamplecmd = '/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/WarpImageMultiTransform 3 bmode.nii.gz bmoderesample.nii.gz identity.txt -R padata1.nii.gz'
maskcmd     = '/opt/apps/itksnap/c3d-1.0.0-Linux-x86_64/bin/c3d -verbose  bmoderesample.nii.gz -thresh 50 inf 1 0 -connected-components -threshold 1 1 1 0 -dilate 1 2x2x2vox -erode 1 2x2x2vox  -o bmodemask.nii.gz'
segmentationcmd = '/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/Atropos -d 3 -a bmoderesample.nii.gz  -x bmodemask.nii.gz -i kmeans[3]  -c [3,0.0] -m [0.1,1x1x1] -k Gaussian -o [bmode_ATROPOS_GMM.nii.gz,bmode_ATROPOS_GMM_POSTERIORS%d.nii.gz]'


