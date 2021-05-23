from Bio.Blast import NCBIXML

result_handle = open("ans/data.xml")
blast_record = NCBIXML.read(result_handle)

sc = blast_record.descriptions
for ind, val in enumerate(blast_record.alignments[:10]):
    lent = val.hsps[0].align_length
    q = sc[ind].score
    iden = val.hsps[0].identities
    gap = val.hsps[0].gaps
    pos = val.hsps[0].positives
    print("%s %d %.2f %.2f %.2f " % (val.accession, q, iden / lent * 100, pos / lent * 100, gap / lent * 100))
