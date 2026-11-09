"""Create isolated report adapters before any new score/image is produced."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'seam_quality_four_v1'

def main():
    s=(OLD/'render.py').read_text(encoding='utf-8')
    s=s.replace("LABELS=('SEAM200',*C.ARMS)", 'LABELS=C.ARMS')
    s=s.replace("srcroot=BASE if label=='SEAM200' else root;arm='SEAM_SLOT_C200' if label=='SEAM200' else label", 'srcroot=root;arm=label')
    s=s.replace('four seen TRAIN conditions', 'four TRAIN / three DEV conditions')
    s=s.replace('seam_quality_four_paired', 'seam_abc_paired')
    s=s.replace('len(jobs)!=32', 'len(jobs)!=56').replace('for x in jobs})!=32','for x in jobs})!=56')
    s=s.replace('Exactly 32 unique', 'Exactly 56 unique')
    s=s.replace('All 128 source cells', 'All 224 source cells')
    s=s.replace('registered_total=32,remaining=31', 'registered_total=56,remaining=55')
    s=s.replace('registered=32', 'registered=56').replace('pages=32','pages=56')
    s=s.replace('SEAM quality four: full fixed panel', 'SEAM A/B/C: full fixed panel')
    s=s.replace('Four seen TRAIN C.', 'Four TRAIN C and three historical DEV C.')
    p=HERE/'render.py'
    if p.exists():raise FileExistsError(p)
    p.write_text(s,encoding='utf-8',newline='\n')

if __name__=='__main__':main()
