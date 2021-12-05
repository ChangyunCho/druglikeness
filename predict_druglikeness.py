from myutils.graph_utils import FILTERS
from myutils.moldesc import adjacency_to_distance, zero_pad
from copy import deepcopy
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MolSurf
from rdkit.Chem import Descriptors

from keras_dgl import MultiGraphCNN
import tensorflow as tf

from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec

electronegativity = {'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04,   
                     'O': 3.44, 'F': 3.98, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90,   
                     'P': 2.19, 'S': 2.59, 'Cl': 3.16, 'K': 0.82, 'Ca': 1.00, 'Sc': 1.36,
                     'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88,
                     'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18,   
                     'Se': 2.55, 'Br': 2.96, 'Kr': 3.00, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22,
                     'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 
                     'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96, 'Sb': 2.05,  
                     'Te': 2.1, 'I': 2.66, 'Xe': 2.6, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10,
                     'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Sm': 1.17, 'Gd': 1.20, 'Dy': 1.22, 
                     'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Lu': 1.27, 'Hf': 1.3, 'Ta': 1.5,
                     'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.20, 'Pt': 2.28, 'Au': 2.54,
                     'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.0, 'At': 2.2,  
                     'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3, 'Pa': 1.5, 'U': 1.38,  'Np': 1.36,   
                     'Pu': 1.28, 'Am': 1.3, 'Cm': 1.3, 'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3,
                     'Fm': 1.3, 'Md': 1.3}

def process_smiles_data(smiles):      
    processor = _smiles_data_processor
   
   #====== processing =======
    tensor_dict = processor(smiles)
   
   #====== joining and padding ============
    max_len_in_data = max(tensor_dict['L'])
    max_len = 128
    F = tensor_dict['F']
    lens = tensor_dict['L']

    X = [zero_pad(x, (max_len, F)) for x in tensor_dict['X']]
    A = [zero_pad(x, (max_len, max_len)) for x in tensor_dict['A']]
    D = [zero_pad(x, (max_len, max_len)) for x in tensor_dict['D']]
   
    result = {}
    result['X'] = np.array(X)
    result['A'] = np.array(A)
    result['D'] = np.array(D)
    result['L'] = np.array(lens)
   
    filter_type = 'first_order'
    adjacency_normalization = True
    filter_generator = FILTERS[filter_type]
    filters = filter_generator(result['A'], result['D'], adjacency_normalization)
   
   #order: X_input, filters_input, nums_input, identity_input, adjacency_input
    res = []
    res.append(result['X'])
    res.append(filters)
    res.append(result['L'])
    res.append(filters[:, :max_len, :])
    res.append(result['A'])
    
    return res

def process_smiles(smiles, use_bond_orders=False, use_formal_charge=False, add_connections_to_aromatic_rings=False, use_Gasteiger=True):
    mol = Chem.MolFromSmiles(smiles)
    A  = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(float)
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, 200, True)
    desc = [describe_atom(x, use_formal_charge=False, use_Gasteiger=True) for x in mol.GetAtoms()]
    
    return np.array(desc), A

def _smiles_data_processor(smiles):
    total_X, total_A, total_lens, total_D = [], [] , [], []
    X, A = process_smiles(smiles)
    D = adjacency_to_distance(A, max_dist_to_include=10)
    n,f = X.shape
    total_lens.append(n)
    total_X.append(X)
    total_A.append(A)
    total_D.append(D)
    
    return {'X':total_X, 'A':total_A, 'L':total_lens, 'F':f, 'D':total_D}

def describe_atom(atom_object, use_formal_charge=False, use_Gasteiger=False):
    p_table = Chem.GetPeriodicTable()
    mol = atom_object.GetOwningMol()
    contribs = MolSurf._LabuteHelper(mol)
    idx = atom_object.GetIdx()
    code = {'SP':1, 'SP2':2, 'SP3':3,'UNSPECIFIED':-1, 'UNKNOWN':-1, 'S':0, 'SP3D':4, 'SP3D2':5}
    result = []
    symbol = atom_object.GetSymbol()
    result.append(atom_object.GetAtomicNum())
    try:
        one_hot = [0.0 for _ in range(7)]
        hib = code[atom_object.GetHybridization().name]
        one_hot[hib+1]=1.0
        #result+=one_hot
        result.append(hib)
        result.append(atom_object.GetTotalValence())
    except:
        print(Chem.MolToSmiles(mol, canonical=0),idx)
        raise
    result.append(max(atom_object.GetNumImplicitHs(), atom_object.GetNumExplicitHs()))
    result.append(p_table.GetNOuterElecs(symbol))
    result.append(electronegativity.get(symbol,0))
    result.append(float(atom_object.GetIsAromatic()))
    if use_formal_charge:
        result.append(atom_object.GetFormalCharge())
    if use_Gasteiger:
        q_in_neu = atom_object.GetDoubleProp('_GasteigerHCharge') + atom_object.GetDoubleProp('_GasteigerCharge')
        result.append(q_in_neu)
    result.append(contribs[idx+1])
    
    return result

def predict_druglikeness(smiles):
    res = process_smiles_data(smiles)
    #load_pretrained classifier
    gcnn_model = tf.keras.models.load_model('./model/gcnn_zinc_trained_model', custom_objects={'MultiGraphCNN':MultiGraphCNN})
    #node feature, filter, num_sample, identity matrix
    temp = [res[0], res[1], res[2], res[3]]
    res = gcnn_model.predict(temp)
    druglikeness_probability = res[0][1]
    
    return druglikeness_probability




