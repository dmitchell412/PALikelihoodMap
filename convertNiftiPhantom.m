clear all
close all

% load example data
exampledata = load('PhantomData.mat')

% get spacing parameters
% FIXME - hack - spacing error ? 
PASpacing = [ exampledata.Ax(2) -   exampledata.Ax(1),...
              exampledata.Lat(2) -   exampledata.Lat(1),...
              1. ]

BMSpacing = [ exampledata.Ax(2) -   exampledata.Ax(1),...
              exampledata.Lat(2) -   exampledata.Lat(1),...
              1. ]

PAExtent = [size(exampledata.PA   ,1) * PASpacing(1), size(exampledata.PA   ,2) * PASpacing(2), size(exampledata.PA   ,3) * PASpacing(3)]
BMExtent = [size(exampledata.Bmode,1) * BMSpacing(1), size(exampledata.Bmode,2) * BMSpacing(2), size(exampledata.Bmode,3) * BMSpacing(3)]

%save bmode data
bmodenii = make_nii(exampledata.Bmode,BMSpacing,[],[],'bmodedata');
save_nii(bmodenii,'PhantomBmode.nii.gz') ;
savevtkcmd  = 'c3d  PhantomBmode.nii.gz -o PhantomBmode.vtk';
disp(savevtkcmd);

%save pa data at each wavelength
%backgroundshift = [ 66.77914    , 55.57656    , 53.79084    , 70.75279    , 80.28145    , 75.00383   ];
backgroundshift = zeros(1,size(exampledata.Wavelength,2));


for iii = 1:size(exampledata.Wavelength,2)
  padatanii = make_nii(exampledata.PA(:,:,iii) - backgroundshift(iii) ,PASpacing);
  save_nii(padatanii,['PhantomPadata.' sprintf('%04d',iii) '.nii.gz']) ;
  savevtkcmd = ['c3d  PhantomPadata.' sprintf('%04d',iii) '.nii.gz PhantomMask.nii.gz -multiply -o PhantomPadata.' sprintf('%04d',iii) '.vtk; sed -i ''s/scalars/padata/g'' PhantomPadata.' sprintf('%04d',iii) '.vtk'];
  disp(savevtkcmd);
  tumorsegmentationcmd = ['/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/Atropos -d 3 -a PhantomPadata.' sprintf('%04d',iii) '.nii.gz  -x PhantomMask.nii.gz -i kmeans[5]  -c [3,0.0] -m [0.1,1x1x1] -k Gaussian -o [Phantom_ATROPOS_GMM.' sprintf('%04d',iii) '.nii.gz,Phantom_ATROPOS_GMM_POSTERIORS%d.' sprintf('%04d',iii) '.nii.gz]'];
  disp(tumorsegmentationcmd);
end

resamplecmd = '/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/WarpImageMultiTransform 3 PhantomBmode.nii.gz bmoderesample.nii.gz offset.txt -R PhantomPadata.0001.nii.gz';
savevtkcmd  = 'c3d  bmoderesample.nii.gz -o bmoderesample.vtk; sed -i ''s/scalars/bmode/g'' bmoderesample.vtk';
maskcmd     = '/opt/apps/itksnap/c3d-1.0.0-Linux-x86_64/bin/c3d -verbose  bmoderesample.nii.gz -thresh 50 inf 1 0 -connected-components -threshold 1 1 1 0 -dilate 1 2x2x2vox -erode 1 2x2x2vox  -o bmodemask.nii.gz';
segmentationcmd = '/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/Atropos -d 3 -a bmoderesample.nii.gz  -x bmodemask.nii.gz -i kmeans[3]  -c [3,0.0] -m [0.1,1x1x1] -k Gaussian -o [bmode_ATROPOS_GMM.nii.gz,bmode_ATROPOS_GMM_POSTERIORS%d.nii.gz]';
PhantomMask   = '/opt/apps/itksnap/c3d-1.0.0-Linux-x86_64/bin/c3d -verbose  bmode_ATROPOS_GMM.nii.gz -thresh 1 1 1 0 -connected-components -threshold 1 1 1 0  -dilate 1 5x5x5vox -o PhantomMask.nii.gz';
tumorsegmentationcmd = ['/opt/apps/ANTsR/dev//ANTsR_src/ANTsR/src/ANTS/ANTS-build//bin/Atropos -d 3  -a PhantomPadata.0001.nii.gz  -a PhantomPadata.0002.nii.gz  -a PhantomPadata.0003.nii.gz  -a PhantomPadata.0004.nii.gz  -a PhantomPadata.0005.nii.gz  -a PhantomPadata.0006.nii.gz  -a PhantomPadata.0007.nii.gz -a PhantomPadata.0008.nii.gz  -x PhantomMask.nii.gz -i kmeans[5]  -c [3,0.0] -m [0.1,1x1x1] -k Gaussian -o [Phantom_ATROPOS_GMM.nii.gz,Phantom_ATROPOS_GMM_POSTERIORS%d.nii.gz]'];
disp(tumorsegmentationcmd);
disp(resamplecmd ); disp(savevtkcmd  ); disp(maskcmd ); disp(PhantomMask  );
