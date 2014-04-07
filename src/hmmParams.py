
# return labeled data as sequences
def getSequences(data, params, rated=False):
   data = data.groupby(['label'])
   seqs = list()
      
   for name,group in data:
      vals = list()
      for p in params:
          vals.append(group[p].values.tolist())
      
      features = map(None,*vals)
      
      if(rated):
          ratings = group['rating'].values.tolist()
          features = map(None,*[features,ratings])
      
      seqs.append(features)

   return seqs

# return HMM parameters init, trans
def hmmMlParams(data,stateAlphabet):

   st = getSequences(data,['rating'])

   # initialize matrices
   states_count = {a: 0 for a in stateAlphabet}
   trans_abs = {a: 0 for a in stateAlphabet}
   trans_ind = {a: {} for a in stateAlphabet}
   transitions = {a: {} for a in stateAlphabet}
   init = {a: 0 for a in stateAlphabet}
   for s in stateAlphabet:
      trans_ind[s] = {a: 0 for a in stateAlphabet}
      transitions[s] = {a: 0 for a in stateAlphabet}

   # for each state sequence
   for k,seq in enumerate(st):
      # for each state transition
      for i, state in enumerate(seq):
          # count number of state
          states_count[state] += 1
          # count number of occurences for initialization            
          if(i==0):
              init[state] += 1
          # count absolute transitions
          if(i<len(seq)-1):
              # inc count of all transitions from this state
              trans_abs[state] += 1
              # inc count of transitions from this state to next one
              trans_ind[state][seq[i+1]] += 1
              

   # divide relative transitions s1->s2 by absolute s1->all
   for s1 in stateAlphabet:
      for s2 in stateAlphabet:
         transitions[s1][s2] = float(
                   trans_ind[s1][s2]) / float(trans_abs[s1])
          
   # divide number of state init occ. by number of seq
   init = {state:float(init[state])/float(len(st)) for state in init}

   transMat = [x.values() for x in transitions.values()]

   return init.values(),transMat
