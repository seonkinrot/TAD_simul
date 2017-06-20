"""
Authors: Seon Ksdsad, Bogdan Bintu
This are the functions for TAD simulations of whole chromosomes in a cell
"""
#Dependencies
import numpy as np
import cPickle as pickle
import glob
import itertools
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import cPickle as pickle

def imshow3d(Im, axis=0, **kwargs):
    """
    Display a 3d ndarray with a slider to move along the 0th dimension.
    Extra keyword arguments are passed to imshow
    """
    im = np.array(Im)
    # generate figure
    f, ax = plt.subplots()
    f.subplots_adjust(left=0.25, bottom=0.25)
    # select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in xrange(3)]
    im_ = im[s].squeeze()
    # display image
    l = ax.imshow(im_, **kwargs)
    l.set_clim(vmin=np.min(im),vmax=np.max(im))
    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax = f.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    slider = Slider(ax, 'Axis %i index' % axis, 0, im.shape[axis] - 1,
                    valinit=0, valfmt='%i')
    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in xrange(3)]
        im_ = im[s].squeeze()
        l.set_data(im_)
        f.canvas.draw()
    slider.on_changed(update)
    plt.show()

def chromosomes(nuc_dia=10000,pixel_sz=100,nchr=46,return_im=False,plt_val=False,cutoff=0.):
    """
    nuc_dia is the nuclear diameter in nm
    pixel_sz is in nm
    cutoff is the extent of inter-chromosome invasion in nm
    This assumes 23 or 46 chromosomes (nchr), including homologues
    
    Return list of chromosomes and pixels in their territory (x,y,z locations in unit radius)
    """
    #coordinates for n sphere centers
    #(equal size spheres in a unit sphere)
    #see:https://oeis.org/A084827/a084827.txt
    if nchr==23:
        centers=[[-0.090186232268855, -0.12719258042466, -0.707348165072374],
                 [0.38870328320077, 0.123439845780586, -0.598602787948567],
                 [-0.309170583212229, 0.37423843622416, -0.537598436535947],
                 [0.28371423154274, -0.462489029599172, -0.479857338821224],
                 [-0.569458339923172, -0.101071012738899, -0.436067408909527],
                 [-0.253827307361331, -0.551289599956929, -0.39535352616894],
                 [0.176894201455841, 0.586308932531292, -0.386785745293835],
                 [0.669516873175885, -0.164919942120003, -0.22181412232117],
                 [0.60918852030665, 0.373364422255918, -0.118917067887175],
                 [-0.269000949817199, 0.668540524640236, -0.073109449104701],
                 [-0.664012065947263, 0.28441188784869, -0.053392789438103],
                 [0.0, 0.0, 0.0],
                 [-0.593650302641085, -0.414670407675602, 0.016754307192129],
                 [0.434917326049123, -0.576009620033541, 0.060932526357327],
                 [-0.09787458412142, -0.709313014594744, 0.109309417466038],
                 [0.232664986830043, 0.668063108437104, 0.155600844104138],
                 [0.631764146516482, -0.135323377713092, 0.327436378946828],
                 [-0.554830942730164, 0.100304097116244, 0.454702813635502],
                 [-0.191032319228568, 0.514177446849921, 0.473055076440126],
                 [-0.332723553965536, -0.40207224871969, 0.502280677592942],
                 [0.216488806207223, -0.4259266153066, 0.544400634457405],
                 [0.373183313581025, 0.301461270612607, 0.542686145412344],
                 [-0.075730170612561, 0.034661741876543, 0.719525149349393]]
    elif nchr==46:
        centers=[[-0.127724638717686,0.029283782782012,-0.763670872459570], 
                [0.302116854275886,0.146601789724809,-0.698281876003332], 
                [0.050116071438789,-0.375084565347080,-0.676139240788969], 
                [0.387404648096449,-0.300279722464142,-0.600095035492607],
                [-0.221565702064757,0.438003368581342,-0.599521487098418],
                [-0.536838502467010,0.121012629438513,-0.545458207564384],
                [0.470578557122151,-0.324839673302964,-0.522876020618661],
                [0.206821475639773,0.544767478949537,-0.510703040725137],
                [0.647737208453552,0.075787428022586,-0.418398254359731],
                [0.209291510617636,-0.653452063989750,-0.359946924370349],
                [-0.240428762326608,-0.655246890184877,-0.336466711372591],
                [0.027563278735129,0.169874066797150,-0.337139524778479],
                [-0.531122333361574,0.491550397468556,-0.276860250786947],
                [-0.125040038594464,0.718782537235944,-0.260923317520113],
                [-0.028222635427186,-0.267579430698296,-0.245896798982907],
                [0.559897837805783,0.479367416697336,-0.238925962888257],
                [-0.609344934400770,-0.421155893776354,-0.227356083644822],
                [-0.755792906627536,0.000918343779410,-0.170705973387576],
                [0.709453517788630,-0.276107684781292,-0.144237918782831],
                [0.338406350902039,-0.029318746498438,-0.079260341210368],
                [0.256184770042010,0.730938689442354,-0.021501641508632],
                [-0.268046158037773,0.223179830668424,-0.001615424109930],
                [0.463839024087979,-0.620577043697123,0.010090454994701],
                [0.761425580114896,0.142996856131315,0.012137124700828],
                [0.041055031342583,-0.772687639260906,0.040405708106847],
                [-0.343201070932800,-0.214763803705687,0.071596445689072],
                [-0.392969757022585,-0.662069840802751,0.087193008193199],
                [-0.377886422912343,0.667723934061050,0.108217022567140],
                [-0.686352373667351,0.339757482368351,0.117684310970756],
                [0.150619047600183,0.321066162828993,0.132327016008240],
                [0.137964450619487,-0.350718167453077,0.164313718413543],
                [0.559387984377712,0.492787670746059,0.211210130456054],
                [-0.717576062734593,-0.078536494382680,0.281568709115817],
                [0.643403410008865,-0.310581345960640,0.299892559603968],
                [0.002276767510746,0.692083481917933,0.348395549284496],
                [-0.069193117297735,-0.000826838519097,0.357871631431749],
                [-0.074584688024342,-0.626168415760149,0.450238341469810],
                [0.622296753862575,0.114447785021264,0.447227819362128],
                [-0.471682318226388,-0.413806749821993,0.454581223971127],
                [-0.434951569989064,0.423001400164857,0.481924550044267],
                [0.305007962363991,-0.417373667885278,0.577177346197253],
                [0.295340191120549,0.432541638100190,0.571004577504622],
                [-0.446844519125231,-0.001070128504388,0.633003282191707],
                [-0.094303907267779,-0.267030401770297,0.721225250748454],
                [0.281485705865138,0.008444506916010,0.721844036069634],
                [-0.091709170433872,0.260484226789782,0.723948700091940]]
    else:
        assert False

    centers = np.array(centers)
    centers = centers[np.random.permutation(len(centers))]
    arr_size = nuc_dia/pixel_sz #division casts as int
    x_ = np.linspace(-1,1,arr_size) #pixel locations in unit radius
    chrters = [[] for i in range(nchr)]
    cutoff_unit_radius = float(cutoff)/(nuc_dia/2.) #convert invasion length to unit radius
    for x in x_:
        for y in x_:
            for z in x_:
                #test if in sphere
                if x*x+y*y+z*z<=1:
                    dists = np.sqrt(np.sum((centers-[[x,y,z]])**2,axis=1)) #distances to all sphere centers
                    dif_2dist = dists-np.min(dists) #how much farther than the closest one?
                    for ichr,dif_dist in enumerate(dif_2dist):
                        if dif_dist<=cutoff_unit_radius:
                            chrters[ichr].append([x,y,z]) #assign pixel to closest chr. and to others within cutoff dist.
    im = np.zeros([arr_size]*3)
    for i,chr_ in enumerate(chrters):
        for x,y,z in (np.array(chr_)+1)*(arr_size-1)/2: #convert to pixel number (1...arr_size)
            if im[int(np.round(x)),int(np.round(y)),int(np.round(z))]>0:
                im[int(np.round(x)),int(np.round(y)),int(np.round(z))]=i+1+len(chrters)
            else:
                im[int(np.round(x)),int(np.round(y)),int(np.round(z))]=i+1
    if plt_val:        
        imshow3d(im,interpolation='nearest')
    if return_im:
        return chrters,im
    return chrters

def TAD_blur(xyzPos,pix_sz=100,nuc_dia=10000): ###add random 3D Gaussian to pixelized TAD location
    perturb=np.random.normal(0,pix_sz/2./(nuc_dia/2.),3)
    return perturb+xyzPos

def TAD_generator(xyzChr,noTADs=100,udist=-0.44276236166846844,sigmadist=0.57416477624326434,nuc_dia=10000,pix_sz=100):
    """
    xyzChr is a list of positions belonging to a chromosome territory
    noTADs is the number of TADs in the chromosome
    udist, sigmadist are the lognormal mean and variance of the distance distribution from consecutive TADs, calculated from
    actual data on chr 21 and 22 (Steven's published data), in units of log(um)
    nuc_dia, pix_sz are the nuclear diamater and pixel size in nm (see above)
    Returns an array of dimensions noTADSx3, representing the 3D location of all TADs in a chromosome
    """
    xyzChr_=np.array(xyzChr)
    tads=[]
    first=xyzChr_[np.random.randint(len(xyzChr))] #randomly choose location of first TAD
    first=TAD_blur(first) #blur so effective resolution is better than pixel
    tads.append(first)
    for i_tad in range(noTADs-1): #sequentially add TADs at distance defined by lognormal distribution
        difs=xyzChr_-[tads[i_tad]]#unit radius
        dists=np.sqrt(np.sum(difs**2,axis=-1))
        dists=np.log(dists*nuc_dia/2000.)#unit log um
        weights = np.exp(-(dists-udist)**2/(2*sigmadist**2))
        weights = np.cumsum(weights)
        weights = weights/float(np.max(weights))
        index_pj = np.sum(np.random.rand()-weights>0)
        pj=xyzChr_[index_pj]#unit radius
        pj=TAD_blur(pj)
        tads.append(pj)
    return np.array(tads)
    
#Encoder - construct a matrix hybes of length number of hybes x number of chromosomes 
#each containing the id of the tad in the hybe (0 means the TAD is missing from that hybe).
#This effectively constructs a code identifying each chromosome
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
    
def test_code(codes):
    """If chromosme i apears in a subset of hybes. Check to see that no other chromosomes appears in the same set."""
    nchr = codes.shape[-1]
    print "No. of tads:"
    print np.unique(np.sum(codes,axis=0))
    print "No. of chrms labeled/hybe:"
    print np.unique(np.sum(codes,axis=1)),np.mean(np.sum(codes,axis=1)),np.std(np.sum(codes,axis=1))
    unique_encoding = np.prod([np.sum(np.prod(codes[codes[:,ichr]==1,:],axis=0))==1 for ichr in range(nchr)])==1
    return unique_encoding
def patch_code(codes,target):
    nchr = codes.shape[-1]
    for ichr in range(nchr):
        code = codes[:,ichr]
        n1s = np.sum(code)
        ndel1s = n1s-target
        if ndel1s>0:
            pos1s = np.where(code)[0]
            del_pos = np.random.choice(pos1s,size=ndel1s,replace=False)
            code[del_pos]=0
        elif ndel1s<0:
            pos0s = np.where(code==0)[0]
            del_pos = np.random.choice(pos0s,size=np.abs(ndel1s),replace=False)
            code[del_pos]=1
    return codes
    
def code_encoder(nchr=23,ntads=100,nlabel_=2,no_hom=1):
    """Master function for the encoder
    nchr is the number of *unique* - i.e. non-homologous - chromosomes
    no_hom is the number of homologous chromosomes
    nlabel is the numbr of TADs labeled in each hybe
    #Interpretation of codes: codes is number of hybe x number of chromosomes and indicates which chr is present in each hybe

    #Interpretation of hybes: hybes is number of hybe x number of chromosomes and indicates which TAD is present in each hybe
    #                         0 means chromose not appearing and if not 0 then it encodes which TAD from the chr appears
    Return hybes
    
    ###Example use:
    hybes = code_encoder(nchr=23,ntads=100,nlabel_=10)
    """
    combs = list(itertools.combinations(range(nchr),nlabel_))
    nhybes = int(float(nchr)*ntads/nlabel_)+1
    inds = np.array(np.round(np.linspace(0,len(combs)-1,nhybes)),dtype=int)
    combs_eq_sp = [combs[ind] for ind in inds]
    codes = combs_to_code(combs_eq_sp)
    codes = patch_code(codes,target=ntads)
    assert test_code(codes)
    hybes=np.cumsum(codes,axis=0)*codes
    hybes=np.concatenate([hybes.T]*no_hom).T
    return hybes
    
def partition_map(list_,map_):
    """
    Inputs
    takes a list [e1,e2,e3,e4,e5,e6] and a map (a list of indices [0,0,1,0,1,2]).  map can be a list of symbols too. ['aa','aa','bb','aa','bb','cc']
    Output
    returns a sorted list of lists, e.g. [[e1, e2,e4],[e3,e5],[e6]]
    """
    list__=np.array(list_,dtype=object)
    map__=np.array(map_)
    return [list(list__[map__==element]) for element in np.unique(map__)]
    
def simulated_imdata(hybes,cell,err_rate=0.032504222398951552):
    """
    Inputs:
    hybes is the encoding(see above)
    cell is ground truth for single cell (no_of_chr x no_of_TADS x 3)
    err_rate is the rate at which a TAD is missed (averaged over 4 chromosomes from Steven's published data)
    Returns:
    hybes_points
    a list of dim no_of_hybes with lists of x,y,z points - the simulated imaging data
    tot_ground_truth is the chromosome identity of all imaged points per hybe
    """
    hybes_points,tot_ground_truth=[],[]
    for hybe in hybes:
        chrs_in_hybe = np.where(hybe>0)[0]
        tad_ids_in_hybe = hybe[hybe>0]-1
        hybe_points,ground_truth=[],[]
        for chr_in_hybe,tad_in_hybe in zip(chrs_in_hybe,tad_ids_in_hybe):
            if np.random.rand()>err_rate: #probability of missing a TAD in imaging
                hybe_points.append(cell[chr_in_hybe][tad_in_hybe])
                ground_truth.append(chr_in_hybe)
        hybes_points.append(hybe_points)
        tot_ground_truth.append(ground_truth)
    hybes_points = map(np.array,hybes_points)
    return hybes_points,tot_ground_truth
def flatten(list_):
    return [item for sublist in list_ for item in sublist]
    
def unique_classif(w_matrix,conf=None):
    """
    Given a weight matrix 'w_matrix' and a confidence function (optional) operation on 0th dimention compute the best unique classification
    and return it as pairs.
    """
    if conf is None: #define default confidence funciton as difference between two "best" options
        def conf(list_):
            #given a projection compute the "confidence" for it as the difference between the two top options
            unk = np.sort(list_)
            if len(unk)<2: #return 0 confidence if only 1 option
                return 0.
            else:
                return unk[1]-unk[0]
    weight_matrix=np.array(w_matrix)
    point_ids = np.arange(weight_matrix.shape[0])
    chr_ids = np.arange(weight_matrix.shape[1])

    chr_picks=[]
    while weight_matrix.shape[0]>0 and weight_matrix.shape[1]>0: #keep a list of unassigned points and chrs, assign one every round and remove it
        confs = map(conf,weight_matrix)# list of confidence for the remaining points across the remaining chrs.
        point_ind = np.argmax(confs)# the id of the point with the highest confidence
        chr_ind = np.argmin(weight_matrix[point_ind])  # the id ot the chromosome assiged to the most confident point
        chr_picks.append((point_ids[point_ind],chr_ids[chr_ind])) #keep above pair

        point_ind_keep = np.setdiff1d(np.arange(weight_matrix.shape[0]),[point_ind])
        chr_ind_keep = np.setdiff1d(np.arange(weight_matrix.shape[1]),[chr_ind])
        point_ids = point_ids[point_ind_keep]
        chr_ids = chr_ids[chr_ind_keep]
        weight_matrix = weight_matrix[point_ind_keep,:]
        weight_matrix = weight_matrix[:,chr_ind_keep] #killing rows and columns of assigned pair
    return chr_picks

#Decoder - Given hybes_points and hybes predict chr id
def decoder(hybes_points,hybes,tot_ground_truth,n_chr=23):
    """
    Compares simulated data with encoding scheme to recreate ground truth by computing distances with nearest neighbors in other hybe rounds
    and projecting onto possible encoding scheme space
    Inputs:
    hybe_points - list of #hybes lists with #pts in hybe 3D positions (the imaging data)
    hybes - the encoding scheme. A matrix of #hybes x #chrs containing binary codes for chr identity
    tot_ground_truth - the actual chromosome association of each point in hybes_points (for assessing performance)
    n_chr - bynber of distinct (non-homologous) chrs
    Returns:
    goods, bads - number of correct and incorrect chr assignment made by the decoder (compared to ground truth), respectively
    chromosome_ids_all - list of #hybes lists with #pts entries corresponding to chr identities of each data point
    """
    #What chromosomes appear in which hybe
    possible_chrs_hybes=[]
    for hybe in hybes:
        possible_chrs_hybes.append(np.where(hybe>0)[0]) #list of chrs appearing in each hybe
    goods,bads=0,0 #initialize counters for QA
    chromosome_ids_all = []
    #Iterate through all the points in each hybe. The current hybe is id_ref
    for id_ref in range(len(hybes_points)):
        ###Given id_ref hybe compute the projection space
        hybes_points_ref = hybes_points[id_ref]
        possible_chrs = possible_chrs_hybes[id_ref] #chromosomes appearing in this hybe
        #compute possible projections: possibble chromosome x numbe of hybes - binary
        possible_projections = np.zeros([len(possible_chrs),len(possible_chrs_hybes)],dtype=int)
        for i,chr_T in enumerate(possible_chrs):
            for j,possible_chrs_hybe in enumerate(possible_chrs_hybes):
                possible_projections[i,j]=chr_T in possible_chrs_hybe #matrix showing for each possible chromosome in this hybe, in which hybes it appears overall
        sum_proj = np.array([np.sum(possible_projections,axis=1)]) 
        possible_projections_ = possible_projections*1./sum_proj.T #the normalized projection space
        #Compute projection of nearest neighbor distances onto hybe scheme space
        projections_point = []
        for point in hybes_points_ref: #an imaged TAD in this round
            min_L1_dists=[]#distances to nearest neighbors across hybes for point
            for hybe_point in hybes_points: #go over all hybes
                difs = point - hybe_point
                min_L1_dist = np.min(np.sqrt(np.sum(difs**2,axis=-1))) #N.N. distance
                min_L1_dists.append(min_L1_dist)
            min_L1_dists = np.array(min_L1_dists)#nearest neighbour distance across hybes for point in reference hybe
            projection = np.dot(possible_projections_,min_L1_dists)
            projections_point.append(projection)
        ##After computing #candidate chromosomes x #points in hybe weight matrix (projections_point),
        ##decide on best assigment.
        chr_picks = unique_classif(projections_point,conf=None)
        points_identities,chr_identities = zip(*chr_picks)
        #chr_identities goes from 0 to number of chromosomes in ref hybe, ordered by assignment confidence
        chromosome_ids0 = np.arange(len(points_identities))
        chromosome_ids0[np.array(points_identities)]=np.array(chr_identities)
        chromosome_ids = possible_chrs[chromosome_ids0]%n_chr #homologous chromosomes are ambiguously identified, because their barcodes are identical
        chromosome_ids_all.append(chromosome_ids)
        #chromosome_ids is chromosome prediction (0-22, if n_chr=23) in order of the points in ref hybe.
        #Compare to ground truth calculated during simulation of imaging data.
        non_deg_poss=np.array(tot_ground_truth[id_ref])%n_chr
        good = np.sum(non_deg_poss==chromosome_ids) #up to degeneracy due to homologous chromosomes
        bad = np.sum(non_deg_poss!=chromosome_ids)
        goods+=good
        bads+=bad                     
    return goods,bads,chromosome_ids_all

def refine_decoder(hybes_points,hybes,prev_decoder_output,tot_ground_truth,n_chr=23,noTads=100,fr_nn=0.8):
    """
    Takes decoded data and refines it to reduce errors. Assignment uses a modified weight matrix, based on median nearest neighbor distances
    and the previous (existing) chr assignment
    Inputs:
    hybes_points - the spatial position (imaging data) of imaged TADs in each hybe (see above)
    hybes - the encoding scheme for each chromosome in each hybe (see above)
    prev_decoder_output - list of #hybes lists with #pts entries corresponding to chr identities of each data point from a previous decoding
    tot_ground_truth - the actual chromosome association of each point in hybes_points
    n_chr - number of non-homologous chrs
    noTads - number of TADs in each chr
    fr_nn - the fraction of TADs in each chr to be used as nearest neighbors for the refinement
    Returns
    goods, bads - number of correct and incorrect chr assignment made by the decoder (compared to ground truth), respectively
    chromosome_ids_all - list of #hybes lists with #pts entries corresponding to chr identities of each data point after refinement
    """
    point_col = flatten(hybes_points)
    chr_col = flatten(prev_decoder_output)
    point_part = partition_map(point_col,chr_col)
    #What chromosomes appear in which hybe
    possible_chrs_hybes=[]
    for hybe in hybes:
        possible_chrs_hybes.append(np.where(hybe>0)[0]%n_chr)
    goods,bads=0,0
    chromosome_ids_all = []
    #Iterate through all the points in each hybe. The current hybe is id_ref
    for id_ref in range(len(hybes_points)):
        hybes_points_ref = hybes_points[id_ref]       
        possible_chrs = possible_chrs_hybes[id_ref]
        weight_chr = []
        for point in hybes_points_ref:
            med_L1_dists=[]
            for pos_chr in possible_chrs: #calculate median distance for each chr
                difs = [point] - np.array(point_part[pos_chr],dtype=float)
                dists = np.sqrt(np.sum(difs**2,axis=-1))
                dists = np.sort(dists)[:int(noTads*fr_nn)] #keep fr_nn*noTads nearest distances
                med_L1_dist = np.median(dists)
                med_L1_dists.append(med_L1_dist)
            med_L1_dists = np.array(med_L1_dists) #median distance to nearest neighbors in each chr for each point in hybe
            weight_chr.append(med_L1_dists)
        ##After computing a no of candidate chromosomes x no of points weight matrix, decide on best assigment.
        chr_picks = unique_classif(weight_chr,conf=None)
        points_identities,chr_identities = zip(*chr_picks)
        #chr_identities goes from 0 to number of chromosomes is ref hybe in maximum confidence order
        chromosome_ids0 = np.arange(len(points_identities))
        chromosome_ids0[np.array(points_identities)]=np.array(chr_identities)
        chromosome_ids = possible_chrs[chromosome_ids0]%n_chr
        chromosome_ids_all.append(chromosome_ids)
        #chromosome_ids is chromosome prediction (0-22) in order of the points in ref hybe.
        #Compare to ground truth calculated during simulation of imaging data.
        non_deg_poss=np.array(tot_ground_truth[id_ref])%n_chr
        good = np.sum(non_deg_poss==chromosome_ids) #up to degeneracy due to homologous chromosomes
        bad = np.sum(non_deg_poss!=chromosome_ids)
        goods+=good
        bads+=bad     
    return goods,bads,chromosome_ids_all

def separator(hybes_points,decoded_data,tot_ground_truth,num_chr=23,num_hom=2):
    """
    Takes decoded data and separates copies of the homologous chromosomes
    Inputs:
    hybes_points - see above
    decoded_data - chromosome_ids_all (see above) after refinement, but with degeneracy of homologous chrs
    tot_ground_truth - see above
    num_chr - number of non-homologous chrs
    num_hom - number of homologous copies of each chr
    Returns
    correct,incorrect - number of correct and incorrect assignments, taking into consideration that homologues are indistinguishable
    decoded_data_hom - same as decoded_data, excpt homologous chrs are distinguished from each other (ids of homologues are separated by
    multiples of num_chr. Fpr example, the homologues of chr 0 will be 23, 46 etc.)
    """
    data_pt_list = map(list,hybes_points) #list of x,y,z positions in each hybe
    for ihybe in range(len(data_pt_list)):
        data_pt_list_hybe = data_pt_list[ihybe] #pts in current hybe
        for ipoint in range(len(data_pt_list_hybe)):
            point = data_pt_list_hybe[ipoint]
            point = list(point)+[ihybe,ipoint] #keep pt position, hybe# and point id#
            data_pt_list_hybe[ipoint] = point
        data_pt_list[ihybe] = data_pt_list_hybe #add information about hybe# and pt id# to data points
    #iterate through chromosomes to patition the data
    pts_chrs,chr_hybes = [],[]
    for pts_hybe,chr_hybe in zip(data_pt_list,decoded_data):
        pts_chr_ = partition_map(pts_hybe,chr_hybe)
        chr_hybe_ = np.unique(chr_hybe)
        pts_chrs.extend(pts_chr_)
        chr_hybes.extend(chr_hybe_)
    pts_partitioned = partition_map(pts_chrs,chr_hybes)
    chrs_ids = np.unique(chr_hybes)
    decoded_data_hom = map(np.array,decoded_data)
    #iterate through chromosome ids and separate the homologues of each chr
    for chrs_id,pts_partitioned_ in zip(chrs_ids,pts_partitioned):
        id_hybe_start = np.argsort(map(len,pts_partitioned_))[-1] #start with a hybe containing all copies of the TADs from the current chr
        pts_start = pts_partitioned_[id_hybe_start]
        assert len(pts_start)==num_hom
        pts_tobeasigned = list(pts_partitioned_)
        pts_tobeasigned.pop(id_hybe_start) #don't reassign the initiator
        chr_estim = [[list(val)+[ival]for ival,val in enumerate(pts_partitioned_[id_hybe_start])]]
        #assign the points in the above hybe to separate homologues - these will serve as seeds to segregate the other chrs
        for pts_hybe in pts_tobeasigned: #go over all remaining hybes
            mean_dists_hybe = []
            chr_estim_flat = np.array(flatten(chr_estim))
            split_chr = partition_map(chr_estim_flat,chr_estim_flat[:,-1]) #buckets with already segregated points
            for pt in pts_hybe:
                med_dists = [np.median([np.sqrt(np.sum((pt[:3]-pt_t[:3])**2)) for pt_t in split_]) for split_ in split_chr]
                #med_dists has num of homolog dists - the median distance of the current point to the points assigned to each
                #homologue so far
                med_dists_hybe.append(med_dists)
            picks = unique_classif(med_dists_hybe) #assign each TAD to one of the homologues based on their relative proximities
            chr_estim.append([list(pts_hybe[pick[0]][:])+[pick[1]] for pick in picks]) #add the data point with an extra number (hom#)
        chr_estim_new = chr_estim #variable for refinement loop
        no_iter = 0
        no_max_iter =10
        while True:
            no_iter+=1
            if no_iter>no_max_iter:
                break
            no_flips,chr_estim_new = refine_separator(chr_estim_new) #no. of changes made in refinement and refined separation
            if no_flips==0:
                break #keep going until no more changes are made
        for pt in flatten(chr_estim_new):
            decoded_data_hom[pt[3]][pt[4]]+=pt[-1]*num_chr #modify chr identity matrix to separate homologues by multiples of num_chr
        correct,incorrect = compare_to_truth(decoded_data_hom,tot_ground_truth,num_hom=num_hom,num_chr=num_chr)
        #compare to ground truth, taking into account equivalence of homologous chrs
    return correct,incorrect,decoded_data_hom

def refine_separator(chr_estim): ###weight and classification method is the same as separator (see above), except using complete hom assigments
    """
    Refining function for separation of homologues
    Inputs:
    chr_estim - list of points, each line containing x,y,z position, #hybe, point id# and all previous rounds of hom assigment
    Returns:
    no_flips - the number of changes made during the refinement
    chr_estim_new - same as input, but with an extra data point of the new hom assigment
    """
    chr_estim_flat = np.array(flatten(chr_estim))
    split_chr = partition_map(chr_estim_flat,chr_estim_flat[:,-1])
    chr_estim_new=[]
    for pts_hybe in chr_estim:
        mean_dists_hybe = []
        for pt in pts_hybe:
            med_dists = [np.median([np.sqrt(np.sum((pt[:3]-pt_t[:3])**2)) for pt_t in split_]) for split_ in split_chr]
            #med_dists has num of homolog dists
            med_dists_hybe.append(med_dists)
        picks = unique_classif(med_dists_hybe)
        chr_estim_new.append([list(pts_hybe[pick[0]][:])+[pick[1]] for pick in picks])
    chr_estim_flat = np.array(flatten(chr_estim))
    chr_estim_flat_new = np.array(flatten(chr_estim_new))  
    reind_old = np.argsort(chr_estim_flat[:,0])
    reind_new = np.argsort(chr_estim_flat_new[:,0])
    pos_dif = np.where(chr_estim_flat[reind_old,-1]!=chr_estim_flat_new[reind_new,-1])[0]
    no_flips = len(pos_dif)
    return no_flips,chr_estim_new

def compare_to_truth(decoded_data_hom,tot_ground_truth,num_chr=23,num_hom=2):
    """
    Function for comparing homologue assigment to ground truth
    Inputs:
    decoded_data_hom - chr identities with homs separated
    tot_ground_truth - ground truth of identities (see above)
    num_chr - number of non-homologous chrs
    num_hom - number of homs for each chr
    Returns:
    correct,total-correct - the number of correct assignments and incorrect (complete to ctotal#) assignments
    """
    pairs = []
    for hb in range(len(decoded_data_hom)): #run on all hybes
        pairs.extend(zip(decoded_data_hom[hb],tot_ground_truth[hb]))
    chrs_pairs = partition_map(pairs,[pair[-1]%num_chr for pair in pairs])
    permuts = np.array(list(itertools.permutations(range(num_hom))))
    #since homs are indistinguishable, keep all permutations of them for robust comparison    
    total = np.sum(map(len,tot_ground_truth))
    correct = 0
    for chr_pairs in chrs_pairs:
        chr_pairs = np.array(chr_pairs,dtype=int)
        id_chrs_truth = np.unique(chr_pairs[:,1])
        perm_values = []
        for permut in permuts:
            id_chrs_data = id_chrs_truth[permut]
            perm_values.append(np.sum([np.sum((chr_pairs[:,0]==idd)*(chr_pairs[:,1]==idt)) for idd,idt in zip(id_chrs_data,id_chrs_truth)]))
        correct+=np.max(perm_values) #find the permutation on hom chr id that best matches ground truth and designate it as our interpretation
    return correct,total-correct