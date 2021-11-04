import os
import os.path
import re
import hashlib
import warnings
import sys
import numpy as np
import pickle
import json
import pathlib
import random
import time

from absl import app
from absl import flags
from absl import logging
from string import ascii_uppercase

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from typing import Dict
from alphafold.common import protein,residue_constants
from alphafold.data import pipeline,parsers,templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch,hhblits,jackhmmer
from alphafold.relax import relax
from typing import Mapping, Optional, Sequence


def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

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

def mk_template(a3m_lines, template_paths):
  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=template_paths,
      max_template_date="2100-01-01",
      max_hits=20,
      kalign_binary_path="kalign",
      release_dates_path=None,
      obsolete_pdbs_path=None)

  hhsearch_pdb70_runner = hhsearch.HHSearch(binary_path="hhsearch", databases=[f"{template_paths}/pdb70"])

  hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
  hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
  templates_result = template_featurizer.get_templates(query_sequence=query_sequence,
                                                       query_pdb_code=None,
                                                       query_release_date=None,
                                                       hits=hhsearch_hits)
  return templates_result.features

def set_bfactor(pdb_filename, bfac, idx_res, chains, is_relaxed=False):
  #logging.info("Chains len : %d",len(chains))
  I = open(pdb_filename,"r").readlines()
  O = open(pdb_filename,"w")
  for line in I:
    if line[0:6] == "ATOM  ":
      seq_id = int(line[22:26].strip()) - 1
      #logging.info("Seq_id : %d",seq_id)
      if not is_relaxed:
        seq_id = np.where(idx_res == seq_id)[0][0]
      O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
  O.close()

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20

bfd_database_path="/vast/projects/alphafold/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
mgnify_database_path="/vast/projects/alphafold/databases/mgnify/mgy_clusters.fa"
template_mmcif_dir="/vast/projects/alphafold/databases/pdb_mmcif/mmcif_files"
obsolete_pdbs_path="/vast/projects/alphafold/databases/pdb_mmcif/obsolete.dat"
pdb70_database_path="/vast/projects/alphafold/databases/pdb70/pdb70"
uniclust30_database_path="/vast/projects/alphafold/databases/uniclust30/uniclust30_2018_08/uniclust30_2018_08"
uniref90_database_path="/vast/projects/alphafold/databases/uniref90/uniref90.fasta"

jackhmmer_binary_path="/stornext/HPCScratch/home/iskander.j/myenvs/alphafold_gpu/bin/jackhmmer"
hhblits_binary_path="/stornext/HPCScratch/home/iskander.j/myenvs/alphafold_gpu/bin/hhblits"
hhsearch_binary_path="/stornext/HPCScratch/home/iskander.j/myenvs/alphafold_gpu/bin/hhsearch"


def main(argv):
    input_fasta_path="input/query_colab.fasta"
    output_dir_base="/vast/scratch/users/iskander.j/alphafold/results_query"
    
    jobname="test"
    
    use_amber = False 
    use_templates = True #@param {type:"boolean"}
    homooligomer = 2 #@param [1,2,3,4,5,6,7,8] {type:"raw"}
    small_bfd_database_path=None
    use_small_bfd=False
    mgnify_max_hits= 501
    uniref_max_hits= 10000
    
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if homooligomer > 1:
      if use_amber:
        logging.info("amber disabled: amber is not currently supported for homooligomers")
        use_amber = False
      if use_templates:
        logging.info("templates disabled: templates are not currently supported for homooligomers")
        use_templates = False
    
    with open(f"{jobname}.log", "w") as text_file:
        text_file.write("use_amber=%s\n" % use_amber)
        text_file.write("use_templates=%s\n" % use_templates)
        text_file.write("homooligomer=%s\n" % homooligomer) 
        
    fasta_name=pathlib.Path(input_fasta_path).stem 
    logging.info("Fasta file: %s",fasta_name)
    output_dir = os.path.join(output_dir_base, fasta_name)
    logging.info("Output Directory: %s",output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)
    
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')

    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)
    logging.info("Input Sequence (%d): %s",num_res, input_seqs)
    
    """
    RUN PIPELINE
    Runs alignment tools on the input sequence and creates features.
    """
    logging.info("Running pipeline")
    jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    if use_small_bfd:
      jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path])
    jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path])
    
    
    logging.info("JackHMMER-UniRef90")
    jackhmmer_uniref90_result =jackhmmer_uniref90_runner.query(input_fasta_path)[0]
    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    with open(uniref90_out_path, 'w') as f:
      f.write(jackhmmer_uniref90_result['sto'])
    uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(
        jackhmmer_uniref90_result['sto'])
    
    logging.info("JackHMMER-Mgnify")
    jackhmmer_mgnify_result = jackhmmer_mgnify_runner.query(input_fasta_path)[0]
    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    with open(mgnify_out_path, 'w') as f:
      f.write(jackhmmer_mgnify_result['sto'])
    mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(
        jackhmmer_mgnify_result['sto'])
    
    mgnify_msa = mgnify_msa[:mgnify_max_hits]
    mgnify_deletion_matrix = mgnify_deletion_matrix[:mgnify_max_hits]
    
    logging.info("Convert to a3m")
    uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
        jackhmmer_uniref90_result['sto'], max_sequences=uniref_max_hits)
    
    logging.info("Running HHBLIT Query-BFD")
    hhblits_bfd_uniclust_result = hhblits_bfd_uniclust_runner.query(
      input_fasta_path)
      
    logging.info("Saving bfd_uniclust_hits.a3m")    
    bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
    with open(bfd_out_path, 'w') as f:
        f.write(hhblits_bfd_uniclust_result['a3m'])
    
    bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
      hhblits_bfd_uniclust_result['a3m'])
    
    '''
    bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
    a3m_lines = "".join(open(bfd_out_path,"r").readlines())
    bfd_msa, bfd_deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)
    #msa=(uniref90_msa, bfd_msa, mgnify_msa)
    #deletion_matrix=(uniref90_deletion_matrix,bfd_deletion_matrix,mgnify_deletion_matrix)
    msa=bfd_msa
    deletion_matrix=bfd_deletion_matrix
    #msa = [i for sub in msa for i in sub]
    #deletion_matrix = [i for sub in deletion_matrix for i in sub]
    '''
    msa=(uniref90_msa, bfd_msa, mgnify_msa)
    deletion_matrix=(uniref90_deletion_matrix,bfd_deletion_matrix,mgnify_deletion_matrix)
    msa = [i for sub in msa for i in sub]
    deletion_matrix = [i for sub in deletion_matrix for i in sub]
    
    if use_templates:
        logging.info("Use Templates")
        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date="2019-05-20",
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path="/stornext/HPCScratch/home/iskander.j/myenvs/alphafold_gpu/bin/kalign",
            release_dates_path=None,
            obsolete_pdbs_path=obsolete_pdbs_path)
        logging.info("Running HHSearch Query on PDB70 ")    
        hhsearch_result = hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
        pdb70_out_path = os.path.join(msa_output_dir, 'pdb70_hits.hhr')
        with open(pdb70_out_path, 'w') as f:
            f.write(hhsearch_result)
        hhsearch_hits = parsers.parse_hhr(hhsearch_result)
        logging.info("Find Template")    
        templates_result = template_featurizer.get_templates(
            query_sequence=input_sequence*homooligomer,
            query_pdb_code=None,
            query_release_date=None,
            hits=hhsearch_hits)
    else:
        logging.info("Use Mock Templates")
        templates_result=mk_mock_template(input_sequence*homooligomer)
    
    
    if homooligomer == 1:
      msas = [msa]
      deletion_matrices = [deletion_matrix]
    else:
      logging.info("H=%d",homooligomer)
      # make multiple copies of msa for each copy
      # AAA------
      # ---AAA---
      # ------AAA
      #
      # note: if you concat the sequences (as below), it does NOT work
      # AAAAAAAAA
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
    
    #logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    #logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 len(templates_result['template_domain_names']))
    
    
    features_dict = {**sequence_features, **msa_features, **templates_result}
    features_output_path=os.path.join(output_dir,"features_orig.pkl")
    with open(features_output_path, 'wb') as f:
        pickle.dump(features_dict, f, protocol=4)
    
    features_output_path=os.path.join(output_dir,"features_orig.pkl")
    features_dict = pickle.load(open(features_output_path, 'rb'))
    
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
    
    # Run the models.

    features_output_path=os.path.join(output_dir,f'features_h{homooligomer}.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(features_dict, f, protocol=4)

    
    relaxed_pdbs = {}
    plddts = {}
    
    # Run the models.
    model_runners = {}
    num_ensemble=1
    model_names=["model_1","model_2"]
    for model_name in model_names:
        model_config = config.model_config(model_name)
        model_config.data.eval.num_ensemble = num_ensemble
        model_params = data.get_model_haiku_params(
            model_name=model_name, data_dir="/vast/projects/alphafold/databases")
        model_runner = model.RunModel(model_config, model_params)
        model_runners[model_name] = model_runner
        
    random_seed = random.randrange(sys.maxsize)    
    logging.info("Random seed: %d",random_seed)
    for model_name, model_runner in model_runners.items():
        logging.info('Running model %s', model_name)
    
        processed_feature_dict = model_runner.process_features(
            features_dict, random_seed=random_seed)
        
        prediction_result = model_runner.predict(processed_feature_dict)
       
        # Get mean pLDDT confidence metric.
        plddts[model_name] = np.mean(prediction_result['plddt'])
        # Save the model outputs.
        result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)
    
        unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                                    prediction_result)

        unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}_h{homooligomer}_orig.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(protein.to_pdb(unrelaxed_protein))
        unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}_h{homooligomer}.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(protein.to_pdb(unrelaxed_protein))
        set_bfactor(unrelaxed_pdb_path, prediction_result['plddt'], idx_res, chains)
    
        # Relax the prediction.
        amber_relaxer = relax.AmberRelaxation(
          max_iterations=RELAX_MAX_ITERATIONS,
          tolerance=RELAX_ENERGY_TOLERANCE,
          stiffness=RELAX_STIFFNESS,
          exclude_residues=RELAX_EXCLUDE_RESIDUES,
          max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
        
        relaxed_pdbs[model_name] = relaxed_pdb_str
        
        
        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}_h{homooligomer}_orig.pdb')
        with open(relaxed_output_path, 'w') as f:
          f.write(relaxed_pdb_str)
        relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}_h{homooligomer}.pdb')
        with open(relaxed_output_path, 'w') as f:
          f.write(relaxed_pdb_str)
        set_bfactor(relaxed_output_path, prediction_result['plddt'], idx_res, chains, is_relaxed=True)

    # Rank by pLDDT and write out relaxed PDBs in rank order.
    ranked_order = []
    for idx, (model_name, _) in enumerate(
      sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
          f.write(relaxed_pdbs[model_name])
    
    ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
    with open(ranking_output_path, 'w') as f:
        f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))
              


if __name__ == '__main__':

  app.run(main)