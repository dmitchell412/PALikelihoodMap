# cmd line
# $ /opt/apps/cubit/v13.2/bin/claro -nographics -batch python meshTemplateRobinBCSurfaceSource.py
# 
# from cubit GUI cmd script editor
#   exec(file("file.py"))
cubit.cmd('set developer on')
cubit.cmd('   reset')

#
#"A catheter for the water-cooled diffusing fibre applicator was 
# designed for use
# with 980nm laser energy delivered through a 400 micron core diameter
# fibreoptic with a total outside diameter (including buffer jacket) of
# 700 micron (figure 1). Custom-sized polyester (PET) preshrunk
# heat-shrink tubing was used to create concentric channels for in- and
# outflow of the cooling fluid. The outer channel was sealed at its
# distal end with a sharpened silica tip for easy tissue penetration.
# The proximal end was constructed by bonding Luer-style connectors such
# that in- and outflow channels were isolated. The catheter was designed
# for cooling of the fibreoptic and surrounding tissue by circulation of
# room temperature water at a flow rate of between 15 and 30 ml/min. The
# thickness of the total water path normal to the diffusing element was
# approximately 350 micron. "

# bounding box for ROI (mm)')
cubit.cmd('create brick  width 50 height 60 depth 60')
cubit.cmd('volume 1 name "healthy" ')
idhealthy = cubit.get_id_from_name('healthy')
print "id" ,idhealthy
cubit.cmd('webcut volume %d with plane xplane' % idhealthy)
cubit.cmd('webcut volume %d with plane yplane' % idhealthy)
cubit.cmd('delete volume 2 3')
#   
# create region for laser and catheter tip
#  active part is about 2mm from catheter tip
#  diffusing tip is 10 mm axial
ActiveDistanceFromTip = 2.0 # mm
cubit.cmd('   webcut volume %d with plane zplane offset %f  ' % (idhealthy,-5.0-ActiveDistanceFromTip) )
cubit.cmd('   webcut volume %d with plane zplane offset -5  ' % idhealthy)
cubit.cmd('   webcut volume %d with plane zplane offset  5  ' % idhealthy)
#   
#  create laser in center of bounding box
#       laser params  used
#       
#       height 100mm
outerRadius  = 1.5 / 2.0 #mm
middleRadius = 1.1 / 2.0 #mm
innerRadius  = 0.7 / 2.0 #mm
print "fiber inner middle outer radius = %f %f %f" % ( outerRadius, middleRadius, innerRadius )
# create outer shell       
cubit.cmd('create cylinder radius %f height 100' % outerRadius)
cubit.cmd('webcut volume 1 4 5 6 tool volume 7 ')
cubit.cmd('delete volume 7')
# create division of inflow/outflow
cubit.cmd('create cylinder radius %f  height 100' % middleRadius)
cubit.cmd('webcut volume 8 9 10 11 tool volume 12 ')
cubit.cmd('delete volume 12')
# create inner fiber       
cubit.cmd('create cylinder radius %f  height 100' % innerRadius )
cubit.cmd('webcut volume 13 14 15 16 tool volume 17 ')
cubit.cmd('delete volume 17')

# merge geometry
cubit.cmd('imprint volume 1 4 5 6 8 9 10 11 13 14 15 16 18 19 20 21')
cubit.cmd('merge volume   1 4 5 6 8 9 10 11 13 14 15 16 18 19 20 21')

resolutionID = 'lores'
meshResolutions = {'lores':(1.0 , 3.8 ), 'midres':(0.5 , 3.0 ), 'hires':(0.25, 2.5 )}
(fineSize,coarseSize) = meshResolutions[resolutionID]
# mesh laser tip
cubit.cmd('volume      10 11 15 16 20 21 size %f' % fineSize )
cubit.cmd('mesh volume 10 11 15 16 20 21')
   
# mesh prostate and fiber
cubit.cmd('volume 1 4 5 6 8 9 13 14 18 19 size %f' % coarseSize )
curveBiasTemplate = 'curve %d scheme bias fine size %f coarse size %f start vertex %d'
#volume 1 radial bias
cubit.cmd( curveBiasTemplate  %( 99,fineSize,coarseSize, 53))
cubit.cmd( curveBiasTemplate  %(101,fineSize,coarseSize, 51))
cubit.cmd( curveBiasTemplate  %(100,fineSize,coarseSize, 54))
cubit.cmd( curveBiasTemplate  %(102,fineSize,coarseSize, 52))
#volume 1 axial bias
cubit.cmd( curveBiasTemplate  %( 83,fineSize,coarseSize, 41))
cubit.cmd( curveBiasTemplate  %( 98,fineSize,coarseSize, 52))
cubit.cmd( curveBiasTemplate  %( 96,fineSize,coarseSize, 51))
cubit.cmd( curveBiasTemplate  %( 82,fineSize,coarseSize, 44))
cubit.cmd( curveBiasTemplate  %( 81,fineSize,coarseSize, 43))
cubit.cmd('mesh volume 1')
# volume 8 axial bias
cubit.cmd( curveBiasTemplate  %( 161,fineSize,coarseSize, 85))
cubit.cmd( curveBiasTemplate  %( 163,fineSize,coarseSize, 87))
cubit.cmd('mesh volume 8')
# volume 6 radial bias
cubit.cmd( curveBiasTemplate  %( 131,fineSize,coarseSize, 69))
cubit.cmd( curveBiasTemplate  %( 132,fineSize,coarseSize, 70))
cubit.cmd('mesh volume 6')
# volume 5 radial bias
cubit.cmd( curveBiasTemplate  %( 117,fineSize,coarseSize, 59))
cubit.cmd( curveBiasTemplate  %( 118,fineSize,coarseSize, 60))
cubit.cmd('mesh volume 5')
# volume 4 radial bias
cubit.cmd( curveBiasTemplate  %( 116,fineSize,coarseSize, 62))
cubit.cmd( curveBiasTemplate  %( 115,fineSize,coarseSize, 61))
# volume 4 axial bias
cubit.cmd( curveBiasTemplate  %( 59,fineSize,coarseSize, 25))
cubit.cmd( curveBiasTemplate  %( 60,fineSize,coarseSize, 27))
cubit.cmd( curveBiasTemplate  %( 58,fineSize,coarseSize, 28))
cubit.cmd( curveBiasTemplate  %(114,fineSize,coarseSize, 60))
cubit.cmd( curveBiasTemplate  %(112,fineSize,coarseSize, 59))
cubit.cmd('mesh volume 4')
# volume 9 axial bias
cubit.cmd( curveBiasTemplate  %(177,fineSize,coarseSize, 93))
cubit.cmd( curveBiasTemplate  %(179,fineSize,coarseSize, 95))
cubit.cmd('mesh volume 9')
# volume 13 18 axial bias
cubit.cmd( curveBiasTemplate  %( 84,fineSize,coarseSize, 42))
cubit.cmd( curveBiasTemplate  %(228,fineSize,coarseSize,121))
cubit.cmd( curveBiasTemplate  %(230,fineSize,coarseSize,122))
cubit.cmd('mesh volume 13 18')
# volume 14 19 axial bias
cubit.cmd( curveBiasTemplate  %( 57,fineSize,coarseSize, 26))
cubit.cmd( curveBiasTemplate  %(244,fineSize,coarseSize,129))
cubit.cmd( curveBiasTemplate  %(246,fineSize,coarseSize,130))
cubit.cmd('mesh volume 14 19')
#   
cubit.cmd('group "badhex"  equals quality hex  in volume  all   jacobian high 0.0')
# export mesh in distinct pieces
cubit.cmd('reset genesis')

# reflect to get full mesh
cubit.cmd('volume all copy reflect x')
cubit.cmd('volume all copy reflect y')
cubit.cmd('imprint volume all')
cubit.cmd('merge volume all')


# create second cylinder for mesh indepenent nodeset
# offset by epsilon to avoid divide by zero in source evaluation
epsilon = 0.05
cubit.cmd('create cylinder radius %f height 10' % ( outerRadius - epsilon ))
cubit.cmd('volume      70 size 0.1' )
cubit.cmd('mesh volume      70 ' )

## # bc
cubit.cmd('skin volume all make sideset 4')
cubit.cmd('sideset 4 name "neumann" ')
## cubit.cmd('sideset 2 face in surface 48 49 61 35 36 71 43 44 55 56 remove')
## cubit.cmd('sideset 3         surface 48 49 61 35 36 71 43 44 55 56')
## cubit.cmd('sideset 3 name "cauchy" ')
## #   ##sideset 3 surface 48 49 61 35 36 71 43 44 55 56 69 79
## #   # specify nodeset for applicator
## #   # skin the applicator to remove the boundary from the nodeset
## #   skin volume 8 10 11 16 18 19 24 26 27 32 34 35 make group 3
## #   group 3 name "appface"
## #   group "appnode" add node in face in group 3
cubit.cmd('nodeset 1 volume 8 10 11 13 15 16 18 20 26 28 29 30 32 33 34 36 42 44 45 46 48 49 50 52 58 60 61 62 64 65 66 68 21 37 53 69')
cubit.cmd('nodeset 1 name "dirichlet"')
# mesh dependent node set
cubit.cmd('nodeset 2 surface 90 206 278 350')
cubit.cmd('nodeset 2 name "sources"')
# mesh independent node set
cubit.cmd('nodeset 3 surface 402 ')
cubit.cmd('nodeset 3 name "IndependentSources"')
## #   #nodeset 1 node in group 4 remove
## #   nodeset 2 node in group 4
## #   nodeset 2 surface 60 80 90
## #   nodeset 2 name "appboundarynode"
## # volume
cubit.cmd('block 1 volume 1 4 5 6 9 14 19 22 23 24 25 27 31 35 38 39 40 41 43 47 51 54 55 56 57 59 63 67')
cubit.cmd('block 1 name "prostate"  ')
cubit.cmd('block 2 volume 8 10 13 15 16 18 20 26 28 30 32 33 34 36 42 44 46 48 49 50 52 58 60 62 64 65 66 68 21 37 53 69')
cubit.cmd('block 2 name "catheder"  ')
cubit.cmd('block 3 volume 11 29 45 61')
cubit.cmd('block 3 name "laserTip"  ')
cubit.cmd('block 4 volume 70 ')
cubit.cmd('block 4 name "laserSources"  ')
cubit.cmd('volume all scale 0.001')
cubit.cmd('export mesh "meshTemplateRobinBCSurfaceSource%s.e" overwrite' % resolutionID)
cubit.cmd('export abaqus "meshTemplateRobinBCSurfaceSource%s.inp" overwrite' % resolutionID)
