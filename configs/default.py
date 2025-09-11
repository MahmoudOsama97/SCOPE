import os

pacs_domain_list = ['p', 'a', 'c', 's']
domainNet_domain_list = ['c', 'i', 'p', 'q', 'r', 's']
officehome_domain_list = ['p', 'a', 'c', 'r']
terra_incognita_list = ['100', '38', '43', '46']
vlcs_domain_list = ['v', 'l', 'c', 's']

slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
if slurm_tmpdir is None:
    slurm_tmpdir = '/scratch/st-msheha02-1/mosama97'

log_count_path = slurm_tmpdir + '/FedDG-MoE2/exps/'
pacs_path = slurm_tmpdir + '/FedDG-MoE2/datasets/pacs'
officehome_path = slurm_tmpdir + '/FedDG-MoE2/datasets/OfficeHome'
vlcs_path = slurm_tmpdir + '/FedDG-MoE2/datasets/vlcs'
domainnet_path = slurm_tmpdir + '/FedDG-MoE2/datasets/DomainNet'
terrainc_path = ''