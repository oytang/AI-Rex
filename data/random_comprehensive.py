import argparse
from time import time
time0 = time()

import pandas as pd
import rdchiral.main as rdc

from falsify_data import removemap
from utils import *

# argument parsing
parser = argparse.ArgumentParser()

parser.add_argument('i_start', type=int)
parser.add_argument('i_end', type=int)

args = parser.parse_args()


# dataloader
df = pd.read_csv('cleaned_uspto50k.csv')
rxn_smiles = list(df['rxn_smiles'])
retro_templates = list(df['retro_template'].drop_duplicates())


# false reaction generator
false_rxn_dict = {
    'rxn_smiles': [],
    'label': [],
    'true_rxn_smiles': [],
    'forward_template': []
}
for i in range(len(df[args.i_start:args.i_end])):
    rxn = rxn_smiles[i]
    rxn = removemap(rxn)
    rt, pt = rxn2rtpt(rxn)

    possible_pts = []
    for j in range(len(retro_templates)):
        # skip if the selected template is the one associated with the reaction
        # because this situation has already been taken into consideration
        # in the STRICT template mapping method
        if df.loc[i, 'retro_template'] == retro_templates[j]:
            continue
        rule = reverse_temp(retro_templates[j])
        possible_pts += rdc.rdchiralRunText(rule, rt)

    # remove duplicates from possible_pts (if any)
    possible_pts = list(set(possible_pts))

    # look for false reactions
    for possible_pt in possible_pts:
        # for possible product that is not the recorded product
        # we take it as a negative product
        # and the corresponding reaction as a negaitve sample
        if possible_pt != pt:
            false_rxn = f'{rt}>>{possible_pt}'

            # dump essential info into dict
            false_rxn_dict['rxn_smiles'].append(false_rxn)
            false_rxn_dict['label'].append(0)
            false_rxn_dict['true_rxn_smiles'].append(rxn)
            false_rxn_dict['forward_template'].append(rule)


false_rxn_df = pd.DataFrame(false_rxn_dict)
print(f'{len(false_rxn_df)} false reactions were succesfully generated this time,')
false_rxn_df.drop_duplicates(subset='rxn_smiles', inplace=True)
print(f'{len(false_rxn_df)} our of which are unique.')
false_rxn_df.to_csv(f'negative_random_{args.i_start:05d}_{args.i_end:05d}.csv', index=False)
print(f'In total, {len(false_rxn_df)} false reactions were successfully generated using random template mapping.')


def format_seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

print(f'Time used: {format_seconds_to_hhmmss(time() - time0)}')


