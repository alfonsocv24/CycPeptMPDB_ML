import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import sys

class Encoder:
    all_features = [
        "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex", 
        "qed", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "NumRadicalElectrons", 
        "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge", 
        "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "BCUT2D_MWHI", "BCUT2D_MWLOW", 
        "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", 
        "BCUT2D_MRLOW", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", 
        "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha", "Ipc", 
        "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", 
        "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", 
        "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", 
        "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", 
        "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", 
        "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", 
        "TPSA", "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", 
        "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", 
        "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", 
        "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "FractionCSP3", "HeavyAtomCount", 
        "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
        "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", 
        "NumHDonors", "NumHeteroatoms", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", 
        "NumSaturatedRings", "RingCount", "MolLogP", "MolMR", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", 
        "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", 
        "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", 
        "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", 
        "fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", 
        "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", 
        "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", 
        "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", 
        "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso", 
        "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid", 
        "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", 
        "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", 
        "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea"]
    fix_features = ["MolWt", "MaxPartialCharge", "MinPartialCharge",
                  "TPSA", "NumHAcceptors", "NumHDonors", "RingCount",
                  "MolLogP"]
    rest_features = [
        "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex", 
        "qed", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "NumRadicalElectrons", 
        "MaxAbsPartialCharge", "MinAbsPartialCharge", 
        "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3", "BCUT2D_MWHI", "BCUT2D_MWLOW", 
        "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", 
        "BCUT2D_MRLOW", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", 
        "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha", "Ipc", 
        "Kappa1", "Kappa2", "Kappa3", "LabuteASA", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", 
        "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", 
        "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", 
        "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", 
        "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3", 
        "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "SlogP_VSA9", 
        "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", 
        "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", 
        "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", 
        "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9", "FractionCSP3", "HeavyAtomCount", 
        "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
        "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", 
        "NumHeteroatoms", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", 
        "NumSaturatedRings", "MolMR", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", 
        "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", 
        "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", 
        "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", 
        "fr_allylic_oxid", "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", 
        "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", 
        "fr_ether", "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", 
        "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", 
        "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso", 
        "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid", 
        "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", 
        "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", 
        "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea"] 

    def __init__(self, properties=None, pca_components=None):
        data = pd.read_csv("CycPeptMPDB_Monomer_All.csv")
        self.seq_data = pd.read_csv("CycPeptMPDB_AllPep.csv", header = 0)
        seq_prop_col = self.seq_data.columns.to_list()
        seq_prop_col = seq_prop_col[2:-1]
        self.all_symbols = data["Symbol"].values

        if pca_components is not None:
            properties = self.rest_features
        else:
            properties = self.all_features

        self.features = data[properties].to_numpy()
        self.fix_feat = data[self.fix_features].to_numpy()
        self.seq_properties = self.seq_data[seq_prop_col].to_numpy()
        # Scale features
        self.features = StandardScaler().fit_transform(self.features)
        self.fix_feat = StandardScaler().fit_transform(self.fix_feat)
        self.seq_properties = StandardScaler().fit_transform(self.seq_properties)
         
        if pca_components is not None:
            self.features = PCA(n_components=pca_components).fit_transform(self.features)
            final_feat = []
            for i in range(len(self.features)):
                a = self.fix_feat[i]
                for j in range(len(self.features[i])):
                    a = np.insert(a, 1, self.features[i][j])
                final_feat.append(a)
            final_feat = np.array(final_feat)
        else:
            final_feat = self.features
            
        # Sort amino acids and properties by the length of the amino acid name (descending)
        self.all_symbols, final_feat = zip(*sorted(zip(self.all_symbols, final_feat), key=lambda x: len(x[0]), reverse=True))

        # Convert features to numpy array. It will be easier later on.
        self.final_feat = np.array(final_feat)
        # self.final_seq_feats = 

    # Parses a sequence and returns an array with integers representing amino acids.
    def parse_sequence(self, subsequence):
        if len(subsequence) == 0:
            return []

        for symbol_idx, symbol in enumerate(self.all_symbols):
            if subsequence.startswith(symbol):
                ret = self.parse_sequence(subsequence[len(symbol):])
                if ret is None:
                    continue
                else:
                    return [symbol_idx] + ret
        return None

    def encode(self, sequences, length, stop_signal=True, sequence_properties = False):
        if stop_signal:
            ret = np.zeros((len(sequences), length, self.final_feat.shape[1] + 1))
        else:
            ret = np.zeros((len(sequences), length, self.final_feat.shape[1]))
        lst_seq_prop = []
        for seq_idx, seq in enumerate(sequences):
            sequence = []
            symbols_indices = self.parse_sequence(seq)
            if symbols_indices is None:
                raise Exception(f"Could not parse sequence {seq}")
            else:
                for pos, symbol_idx in enumerate(symbols_indices):
                    ret[seq_idx, pos, :self.final_feat.shape[1]] = self.final_feat[symbol_idx]
                    #print(self.all_symbols[symbol_idx])
                    sequence.append(self.all_symbols[symbol_idx])
            
            if stop_signal:
                ret[seq_idx, pos - 1, -1] = 1
            if sequence_properties:
                idx_feat_seq = self.seq_data[self.seq_data['Sequence'] == seq].index.values
                seq_feat = self.seq_properties[idx_feat_seq]
                # print(seq_feat[0])
                lst_seq_prop.append(seq_feat[0])
        lst_seq_prop = np.array(lst_seq_prop).astype(np.float32)
        if sequence_properties:
            return ret, sequence, lst_seq_prop
        else:
            return ret, sequence




