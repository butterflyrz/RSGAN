import numpy as np
from Models import MF
from Models import FISM

class SaverFactory(object):
    @staticmethod
    def getSaver(model):
        if isinstance(model, MF):
            return MFSaver()
        if isinstance(model, FISM):
            return FISMSaver()
        return Saver()

class Saver(object):
    def save(self, model, sess):
        Null
    def setPrefix(self, prefix = None):
        self.prefix = prefix

class MFSaver(Saver):
    def __init__(self):
        self.prefix = None

    def save(self, model, sess):
        if self.prefix == None:
            print "prefix should be set by Saver.setPrefix(prefix)"
            return

        params = sess.run([model.embedding_P, model.embedding_Q])
        print 'saving model.embedding_P', params[0].shape, ', model.embedding_Q', params[1].shape,\
              ' to', "MF_%s_*.txt" % self.prefix

        f = open('MF_' + self.prefix + "_P.txt", 'w')
        np.savetxt(f, params[0])
        f.close()

        f = open('MF_' + self.prefix + "_Q.txt", 'w')
        np.savetxt(f, params[1])
        f.close()

class FISMSaver(Saver):
    def __init__(self):
        self.prefix = None

    def save(self, model, sess):
        if self.prefix == None:
            print "prefix should be set by Saver.setPrefix(prefix)"
            return
        params = sess.run([model.embedding_P, model.embedding_Q])
        print 'saving model.embedding_P', params[0].shape, ', model.embedding_Q', params[1].shape,\
              ' to',  "FISM_%s.npy" % self.prefix
        np.save(self.prefix + ".npy", {'P': params[0], 'Q': params[1]})

