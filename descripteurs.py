import cv2
from skimage.feature import graycomatrix,graycoprops
from BiT import bio_taxo
from mahotas.features import haralick

import numpy as np

# niveau de gris 
def glcm(chemin):
    data=cv2.imread(chemin,0)
    co_matrice=graycomatrix(data,distances=[1],angles=[3*np.pi/2],symmetric=False,normed=True)
    contrast=graycoprops(co_matrice,'contrast')[0,0]
    dissimilarity=graycoprops(co_matrice,'dissimilarity')[0,0]
    homogeneity=graycoprops(co_matrice,'homogeneity')[0,0]
    correlation=graycoprops(co_matrice,'correlation')[0,0]
    energy=graycoprops(co_matrice,'energy')[0,0]
    ASM=graycoprops(co_matrice,'ASM')[0,0]
    features= [contrast,dissimilarity,homogeneity,correlation,energy,ASM]
    features= [float(x) for x in features]
    return features


def haralick_feat(chemin):
    data=cv2.imread(chemin,0)
    features=haralick(data).mean(0).tolist()
    features=[float(x) for x in features]
    return features


def bitdesc_feat(chemin):
    data=cv2.imread(chemin,0)
    features=bio_taxo(data)
    features=[float(x) for x in features]
    return features


def concatenation(chemin):
    return glcm(chemin)+haralick_feat(chemin)+bitdesc_feat(chemin)



# RGB
def glcm_RGB(chemin):
    data=cv2.imread(chemin)
    data=cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
    list_carac=[]
    for i in range(3):
        canal=data[:,:,i]
        co_matrice=graycomatrix(canal,distances=[1],angles=[3*np.pi/2],symmetric=False,normed=True)
        contrast=graycoprops(co_matrice,'contrast')[0,0]
        dissimilarity=graycoprops(co_matrice,'dissimilarity')[0,0]
        homogeneity=graycoprops(co_matrice,'homogeneity')[0,0]
        correlation=graycoprops(co_matrice,'correlation')[0,0]
        energy=graycoprops(co_matrice,'energy')[0,0]
        ASM=graycoprops(co_matrice,'ASM')[0,0]
        features= [contrast,dissimilarity,homogeneity,correlation,energy,ASM]
        features= [float(x) for x in features]

        list_carac.extend(features)
    return features

def haralick_feat_RGB(chemin):
    data=cv2.imread(chemin)
    data=cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
    list_carac=[]
    for i in range(3):
        canal=data[:,:,i]    
        features=haralick(canal).mean(0).tolist()
        features=[float(x) for x in features]
        list_carac.extend(features)
    return list_carac

def bitdesc_feat_RGB(chemin):
    data=cv2.imread(chemin)
    data=cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
    list_carac=[]
    for i in range(3):
        canal=data[:,:,i]
        features=bio_taxo(canal)
        features=[float(x) for x in features]
        list_carac.extend(features)
    return list_carac

def concatenation_RGB(chemin):
    return glcm_RGB(chemin)+haralick_feat_RGB(chemin)+bitdesc_feat_RGB(chemin)

def glcm_concat_RGB_NG(chemin):
    return glcm_RGB(chemin)+glcm(chemin)


