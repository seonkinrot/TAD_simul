import itertools
import numpy as np
from scipy.spatial.distance import pdist,cdist
def combs_to_code(combs_eq_sp,nchr=None):
    """Construct from combs list to code binary array
    For example changes:
    combs_eq_sp = [(0,1,2,3),(0,1,2,4)] to codes = [[1,1,1,1,0],[1,1,1,0,1]]
    """
    if nchr is None:
        nchr = np.max(combs_eq_sp)+1
    codes = np.zeros([len(combs_eq_sp),nchr],dtype=int)
    for i,comb in enumerate(combs_eq_sp):
        codes[i][list(comb)] = 1
    return codes
def hamming_distance(codei,codej):
    return np.sum(codei!=codej)
def simulate_codes(all_codes,dst_min=6):
    codes=[]
    keep_codes = np.array(all_codes)
    while len(keep_codes)>0:
        #print len(codes)
        ich = np.random.randint(0,len(keep_codes))
        codes.append(keep_codes[ich])
        keep_codes_=[]
        for code_i in keep_codes:
            Good=True
            for code_j in codes:
                if hamming_distance(code_i,code_j)<dst_min:
                    Good=False
                    break
            if Good:
                keep_codes_.append(code_i)
        keep_codes=keep_codes_
    return np.array(codes)
def simulate_codes_max(all_codes,ncode=23):
    codes=[]
    keep_codes = np.array(all_codes)
    for i in range(ncode):
        if i>0:
            sum_dists = np.min(cdist(codes,keep_codes,'hamming'),axis=0)
            max_pos = np.where(sum_dists==np.max(sum_dists))[0]
            ich = np.random.choice(max_pos)
        else:
            ich = np.random.randint(len(keep_codes))
        codes.append(keep_codes[ich])
    return np.array(codes)
def split_codes(sim_,npair=100000):
    sim1,sim2=sim_[:len(sim_)/2],sim_[len(sim_)/2:]
    sim1_,sim2_=np.array(sim1),np.array(sim2)
    old = np.std(np.sum(sim1_,axis=0))+np.std(np.sum(sim2_,axis=0))

    for i in range(npair):
        i1,i2=np.random.randint(len(sim1_)),np.random.randint(len(sim2_))
        sim1_[i1],sim2_[i2]=sim2[i2],sim1[i1]
        new = np.std(np.sum(sim1_,axis=0))+np.std(np.sum(sim2_,axis=0))
        if new<old:
            old=new
            sim1,sim2=np.array(sim1_),np.array(sim2_)
        else:
            sim1_,sim2_=np.array(sim1),np.array(sim2)
    return sim1,sim2
def multi_codes(nhybes=16,ntads=10,num_codes=46,n_sim=1000,verbose=False):
    if verbose:
        print "Computing combinations..."
    combs = list(itertools.combinations(range(nhybes),ntads))
    code = combs_to_code(combs)
    if verbose:
        print "Computing maximum spearation simulation..."
    sims_ = [] 
    for sim_n in range(n_sim):
        if verbose:
            print sim_n
        sims_.append(simulate_codes_max(code,num_codes))
    return sims_
def pick_best_sim_and_splitin2(sims_,n_split=100000):
    sims_=np.array(sims_)
    min_ham_dists = np.array([np.min(pdist(sim_,'hamming')) for sim_ in sims_])
    print np.max(min_ham_dists)*sim_.shape[-1]
    sim_maxs = sims_[min_ham_dists==np.max(min_ham_dists)]
    std_min = np.argmin([np.std(np.sum(sim_,axis=0)) for sim_ in sim_maxs])
    sim_ = sim_maxs[std_min]
    print np.sum(sim_,axis=0)
    sim1,sim2=split_codes(sim_,n_split)
    print np.sum(sim1,axis=0)
    print np.sum(sim2,axis=0)
    return sim1,sim2
def collisions(sim1,sim2,nTADperChr=1):
    if nTADperChr==1:
        col_mat = np.zeros([len(sim1),len(sim2),len(sim1[0])],dtype=int)
        for i1,sim1_ in enumerate(sim1):
            for i2,sim2_ in enumerate(sim2):
                col_mat[i1,i2] = (sim1_==sim2_)*(sim1_>0)*(sim2_>0)
        return col_mat
    if nTADperChr==2:
        col_mat = np.zeros([len(sim1)/2,len(sim2)/2,len(sim1[0])],dtype=int)
        h1,h2,n_chr = col_mat.shape
        for i1 in range(h1):
            for i2 in range(h2):
                vals = np.zeros(n_chr)
                for sim1_ in [sim1[i1],sim1[i1+h1]]:
                    for sim2_ in [sim2[i2],sim2[i2+h2]]:
                        vals = vals+(sim1_==sim2_)*(sim1_>0)*(sim2_>0)
                col_mat[i1,i2] = vals**10
        return col_mat

def min_collisions(sim1_tad,sim2_tad,niter=1000,nTADperChr=1):
    col_mat = collisions(sim1_tad.T,sim2_tad.T,nTADperChr=nTADperChr)
    col_mat_sum = np.sum(col_mat,axis=-1)
    pairs_colisions = col_mat_sum*(col_mat_sum-1)/2
    prev_col = np.sum(pairs_colisions)
    print "No. col. start:"+str(prev_col)
    x_col,y_col = np.where(pairs_colisions)
    w_col = pairs_colisions[pairs_colisions>0]
    w_col = np.cumsum(w_col)
    w_col = w_col/float(np.max(w_col))
    n_stuck=0
    while n_stuck<niter:
        n_stuck+=1
        index_ch = np.sum(np.random.rand()-w_col>0)
        h1 = x_col[index_ch]
        h2 = y_col[index_ch]
        chrs_col = np.where(col_mat[h1,h2])[0]
        chr_ch = np.random.choice(chrs_col)
        
        h_ch = np.random.choice(np.where(sim2_tad[chr_ch,:])[0])

        sim2_tad_ = np.array(sim2_tad)
        if nTADperChr==2:
            h2 = h2+np.random.choice([0,sim2_tad.shape[-1]/2])
        sim2_tad_[chr_ch,h2],sim2_tad_[chr_ch,h_ch]=sim2_tad[chr_ch,h_ch],sim2_tad[chr_ch,h2]
        col_mat = collisions(sim1_tad.T,sim2_tad_.T,nTADperChr=nTADperChr)
        col_mat_sum = np.sum(col_mat,axis=-1)
        pairs_colisions = col_mat_sum*(col_mat_sum-1)/2
        new_col = np.sum(pairs_colisions)
        if new_col<prev_col:
            x_col,y_col = np.where(pairs_colisions)
            w_col = pairs_colisions[pairs_colisions>0]
            w_col = np.cumsum(w_col)
            w_col = w_col/float(np.max(w_col))
            sim2_tad = np.array(sim2_tad_)
            prev_col = new_col
            n_stuck=0
    print "No. col. end:"+str(prev_col)
    return sim2_tad,prev_col
def multi_min_collisions(sim1,sim2,niter=100,n_iter=1000,nTADperChr=1):
    sim1_tad = np.cumsum(sim1,axis=-1)*sim1
    sim2_tads=[]
    col_ns=[]
    for i in range(niter):
        sim2_tad = np.array(sim2)
        for sim__ in sim2_tad:
            sim__[sim__>0]=np.random.permutation(np.sum(sim__))+1
        sim2_tad,col_n = min_collisions(sim1_tad,sim2_tad,niter=n_iter,nTADperChr=nTADperChr)
        sim2_tads.append(sim2_tad)
        col_ns.append(col_n)
    return sim2_tads,col_ns
