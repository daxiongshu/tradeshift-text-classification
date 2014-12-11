def split_file_by_half(fname,data_dir,data_cache_dir):
    # fname: the name of the source file
    # this function divides the source file into two halfs
    f=open(data_dir+fname+'.csv')
    f.readline()
    lines=0
    for line in f:  
        lines+=1
    f.close()
    f=open(data_dir+fname+'.csv')
    f1=open(data_cache_dir+fname+'_part1.csv','w')
    f2=open(data_cache_dir+fname+'_part2.csv','w')
    head=f.readline()
    f1.write(head)
    f2.write(head)
    
    for c,line in enumerate(f):
        if c<lines/2:
            f1.write(line)
        else:
            f2.write(line)
    f1.close()
    f2.close()
    f.close()


def split(data_dir,data_cache_dir):
# split the files for online models
    split_file_by_half('train',data_dir,data_cache_dir)
    split_file_by_half('trainLabels',data_dir,data_cache_dir)


import gzip
def save_predictions(name, ids, predictions) :
    out = gzip.open(name, 'w')
    print >>out, 'id_label,pred'
    for id, id_predictions in zip(ids, predictions) :
        for y_id, pred in enumerate(id_predictions) :
            if pred == 0 or pred == 1 :
                pred = str(int(pred))
            else :
                pred = '%.6f' % pred
            print >>out, '%d_y%d,%s' % (id, y_id + 1, pred)

if __name__ == "__main__":
    split(data_dir='../../data-sample/', data_cache_dir='./')
