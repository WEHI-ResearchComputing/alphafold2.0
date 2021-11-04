import numpy as np
import os, json, pickle
from absl import logging


# Chains to/from files
def write_chains(chains_path,chains):
    with open(chains_path, 'w') as filehandle:
        filehandle.writelines("%s\n" % chain for chain in chains)
    
def read_chains(chains_path):
    chains = []
    # open file and read the content in a list
    with open(chains_path, 'r') as filehandle:
        filecontents = filehandle.readlines()
        for line in filecontents:
            current_place = line[:-1]
            chains.append(current_place)
    return chains

# Adding confidence measure pddlt to models and saving to file
def set_bfactor(ip_path,op_path, bfac):
    I = open(ip_path,"r").readlines()
    O = open(op_path,"w")
    for line in I:
        if line[0:6] == "ATOM  ":
            seq_id = int(line[22:26].strip()) - 1
            O.write(f"{line[:60]}{bfac[seq_id]:6.2f}{line[66:]}")
    O.close()
    
# Adding confidence measure pddlt to models, reseting chains, for h>1 only,  and saving to file    
def set_chain_bfactor(ip_path,op_path, bfac,  chains,idx_res=None, is_relaxed=False):

    #logging.info("Chains len : %d",len(chains))
    I = open(ip_path,"r").readlines()
    O = open(op_path,"w")
    for line in I:
        if line[0:6] == "ATOM  ":
          seq_id = int(line[22:26].strip()) - 1
          #logging.info("Seq_id : %d",seq_id)
          if not is_relaxed:
            seq_id = np.where(idx_res == seq_id)[0][0]
          O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
    O.close()
    
##Reranking
def rerank(output_dir):
    plddts={}
    models=['model_1', 'model_2','model_3','model_4','model_5']
    for model_name in models:
        m_path=os.path.join(output_dir,f'result_{model_name}.pkl')
       
        if os.path.exists(m_path):
            logging.info(f"Found results for {model_name}")
            result=pickle.load(open(m_path, 'rb'))
            plddts[model_name] = np.mean(result['plddt'])
        else:
            logging.info(f"No results found for {model_name},{m_path}")
    ranked_order = []
    logging.info("Ordering and saving.")
    for idx, (model_name, _) in enumerate(sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
    
    ranking_output_path = os.path.join(output_dir, 're-ranking_debug.json')
    with open(ranking_output_path, 'w') as f:
        f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))
    
    
    