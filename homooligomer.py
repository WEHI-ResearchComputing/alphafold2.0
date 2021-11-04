import numpy as np
import os
from alphafold.data import parsers,templates,pipeline
from string import ascii_uppercase
from absl import logging
def mk_mock_template(query_sequence):
  # since alphafold's model requires a template input
  # we create a blank example w/ zero input, confidence -1
  ln = len(query_sequence)
  output_templates_sequence = "-"*ln
  output_confidence_scores = np.full(ln,-1)
  templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
  templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)
  template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                       'template_all_atom_masks': templates_all_atom_masks[None],
                       'template_sequence': [f'none'.encode()],
                       'template_aatype': np.array(templates_aatype)[None],
                       'template_confidence_scores': output_confidence_scores[None],
                       'template_domain_names': [f'none'.encode()],
                       'template_release_date': [f'none'.encode()]}
  return template_features

def load_msas(msa_output_dir,mgnify_max_hits=10000):
    bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
    a3m_lines = "".join(open(bfd_out_path,"r").readlines())
    bfd_msa, bfd_deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)
    jackhmmer_uniref90_result = "".join(open(os.path.join(msa_output_dir,"uniref90_hits.sto"),"r").readlines())
    uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(jackhmmer_uniref90_result)
    jackhmmer_mgnify_result = "".join(open(os.path.join(msa_output_dir,"mgnify_hits.sto"),"r").readlines())
    mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(jackhmmer_mgnify_result)
    mgnify_msa = mgnify_msa[:mgnify_max_hits]
    mgnify_deletion_matrix = mgnify_deletion_matrix[:mgnify_max_hits]
    
    msa=(uniref90_msa, bfd_msa, mgnify_msa)
    deletion_matrix=(uniref90_deletion_matrix,bfd_deletion_matrix,mgnify_deletion_matrix)
    msas = [i for sub in msa for i in sub]
    deletion_matrices = [i for sub in deletion_matrix for i in sub]
    return msas,deletion_matrices


def create_features_with_h(fasta_path, homooligomer,msa,deletion_matrix):
    with open(fasta_path) as f:
      input_fasta_str = f.read()
    
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')

    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    
    msas = []
    deletion_matrices = []
    Ln = len(input_sequence)
    for o in range(homooligomer):
        L = Ln * o
        R = Ln * (homooligomer-(o+1))
        msas.append(["-"*L+seq+"-"*R for seq in msa])
        deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in deletion_matrix])
    
    sequence_features = pipeline.make_sequence_features(
        sequence=input_sequence*homooligomer,
        description=input_description,
        num_res=len(input_sequence)*homooligomer)
        
    msa_features = pipeline.make_msa_features(
        msas=msas,
        deletion_matrices=deletion_matrices)
    
    templates_result=mk_mock_template(input_sequence*homooligomer)
    features_dict = {**sequence_features, **msa_features, **templates_result}
    features_dict,chains=update_residue(features_dict,input_sequence,homooligomer)
    
    return features_dict,chains

def update_residue(features_dict,input_sequence,homooligomer):
    # Minkyung's code
    # add big enough number to residue index to indicate chain breaks
    idx_res = features_dict['residue_index']
    L_prev = 0
    # Ls: number of residues in each chain
    Ls=[len(input_sequence)]*homooligomer
    for L_i in Ls[:-1]:
      idx_res[L_prev+L_i:] += 200
      L_prev += L_i  
    chains = list("".join([ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
    features_dict['residue_index'] = idx_res
    return features_dict,chains
    