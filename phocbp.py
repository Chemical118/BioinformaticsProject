from Bio.Blast import NCBIXML
from Bio.Blast import NCBIWWW

with open('ans/ans.txt', 'r', encoding='utf-8') as f:
    ans_list = f.readlines()
tar = ans_list[-1].split()[-1]
result_handle = NCBIWWW.qblast('blastp', 'nr', tar)

with open('ans/data.xml', 'w') as q:
    q.write(result_handle.read())
result_handle.close()
blast_record = NCBIXML.read(open('ans/data.xml'))

sc = blast_record.descriptions
for ind, val in enumerate(blast_record.alignments[:10]):
    lent = val.hsps[0].align_length
    q = sc[ind].score
    iden = val.hsps[0].identities
    gap = val.hsps[0].gaps
    pos = val.hsps[0].positives
    print("%s %d %.2f %.2f %.2f " % (val.accession, q, iden / lent * 100, pos / lent * 100, gap / lent * 100))
