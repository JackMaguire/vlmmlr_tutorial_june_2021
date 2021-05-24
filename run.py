#import pyrosetta
from pyrosetta import *
pyrosetta.init()

import menten_gcn as mg
import menten_gcn.decorators as decs

from spektral.layers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import numpy as np


decorators = [ decs.SameChain(), decs.CBCB_dist(), decs.PhiPsiRadians() ]

#requires Rosetta:
decorators.extend([ decs.Ref2015Decorator(), decs.RosettaHBondDecorator() ])

xml = '''<RESIDUE_SELECTORS>
<ResiduePropertySelector name="d_amino_acid" properties="D_AA" />
</RESIDUE_SELECTORS>'''
decorators.append( decs.RosettaResidueSelectorFromXML( xml, "d_amino_acid" ) )

data_maker = mg.DataMaker( decorators=decorators,
                           edge_distance_cutoff_A=12.0,
                           max_residues=25 )

data_maker.summary()


Xs = []
As = []
Es = []
outs = []

poses = []
ddgs = []

datafile = 'test_peptides/ddg_data.dat'
#datafile = 'test_peptides/ddg_data.dat.head'

with open( datafile, 'r' ) as f:
    for line in f.readlines():
        tokens = line.split()
        if len( tokens ) == 2:
            filename = "test_peptides/" + tokens[0]
            poses.append( pose_from_pdb( filename ) )
            ddgs.append( float( tokens[1] ) )

assert len(poses) == len(ddgs), "{} {}".format( len(poses), len(ddgs) )

for i in range( 0, len(poses) ):
    print( "i", i )
    pose = poses[i]
    
    wrapped_pose = mg.RosettaPoseWrapper( pose )
    cache = data_maker.make_data_cache( wrapped_pose )

    assert pose.num_chains() == 2
    n_peptide_residues = len( pose.chain_sequence(2) )
    first_peptide_resid = pose.size() - n_peptide_residues + 1

    # poor ahead-of-time curation on my part
    peptide_resids = []
    has_bad_residue = False
    for resid in range( first_peptide_resid, pose.size()+1 ):
        peptide_resids.append( resid )
        if pose.residue(resid).name3() in [ 'DAB', 'ORN', 'DPP' ]:
            has_bad_residue = True
    if has_bad_residue:
        continue

    assert len(peptide_resids) == n_peptide_residues

    X, A, E, resids = data_maker.generate_input( wrapped_pose, focus_resids=peptide_resids, data_cache=cache, sparse=False )
    Xs.append( X )
    As.append( A )
    Es.append( E )

    outs.append([ ddgs[i], ])
    

Xs = np.asarray( Xs )
As = np.asarray( As )
Es = np.asarray( Es )
outs = np.asarray( outs )

print( Xs.shape )
print( As.shape )
print( Es.shape )
print( outs.shape )

X_in, A_in, E_in = data_maker.generate_XAE_input_tensors()
X = ECCConv( 10, activation='relu' )([X_in, A_in, E_in])
X = ECCConv( 10, activation='relu' )([X,    A_in, E_in])
X = GlobalSumPool()(X)
X = Flatten()(X)
output = Dense( 1, name='out' )(X)

model = Model(inputs=[X_in,A_in,E_in], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error' )
model.summary()


model.fit( x=[Xs,As,Es], y=outs, batch_size=32, epochs=100, validation_split=0.2 )
