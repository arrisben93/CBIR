import cv2
from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concatenation_RGB
import os

import numpy as np 

def extraction_signatures(chemin_repertoire):
    list_glcm = []
    list_haralick = []
    list_bit = []
    list_concat = []



    for root, _, files in os.walk(chemin_repertoire):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                relative_path = os.path.relpath(os.path.join(root, file), chemin_repertoire)
                chemin = os.path.join(root, file)
                class_name = os.path.dirname(relative_path)

                try:
                    glcm_feat = glcm_RGB(chemin) + [class_name, relative_path]
                    haralick_feat = haralick_feat_RGB(chemin) + [class_name, relative_path]
                    bit_feat = bitdesc_feat_RGB(chemin) + [class_name, relative_path]
                    concat_feat = concatenation_RGB(chemin) + [class_name, relative_path]

                    list_glcm.append(glcm_feat)
                    list_haralick.append(haralick_feat)
                    list_bit.append(bit_feat)
                    list_concat.append(concat_feat)
                except Exception as e:
                    print(f"Erreur Dans l'extraction de {chemin} : {e}")



        np.save("signatures/signatures_GLCM_RGB.npy", np.array(list_glcm, dtype=object))
        np.save("signatures/signatures_Haralick_RGB.npy", np.array(list_haralick, dtype=object))
        np.save("signatures/signatures_BiT_RGB.npy", np.array(list_bit, dtype=object))
        np.save("signatures/signatures_Concat_RGB.npy", np.array(list_concat, dtype=object))

    print(" Enregistrement des fichiers .npy efféctuer.")

if __name__ == '__main__':
    extraction_signatures('./dataset/')
