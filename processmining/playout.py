import random
from queue import Queue
from copy import copy
from pm4py.objects.petri import semantics

class PotentialTrace():
    def __init__(self, marking, firingSequence, marking_count, inv_counter=0):
        self.marking = marking
        self.firingSequence = firingSequence
        self.inv_counter = inv_counter
        self.marking_count = marking_count

    def getMarking(self):
        return self.marking

    def getMarkingCount(self):
        return self.marking_count

    def getFiringSequence(self):
        return self.firingSequence

    def getInvCounter(self):
        return self.inv_counter

class Player():
    def __init__(self, net, initial_marking, final_marking, maxTraceLength, rep_inv_thresh=100, max_loop=3):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.maxTraceLength = maxTraceLength
        self.rep_inv_thresh = rep_inv_thresh
        self.max_loop = max_loop

    def play(self):
        self.generatedTraces = set()
        self.potentials = Queue()

        marking_count = dict()
        for mark in self.initial_marking:
            marking_count[mark] = 1

        self.potentials.put_nowait(PotentialTrace(marking = copy(self.initial_marking), firingSequence=list(), marking_count=marking_count))

        while not self.potentials.empty():
            potential = self.potentials.get_nowait()
            marking = potential.getMarking()
            firingSeq = potential.getFiringSequence()

            enabled_trans = semantics.enabled_transitions(self.net, marking)
            for enabled_tran in enabled_trans:
                new_marking = semantics.execute(enabled_tran, self.net, marking)
                new_firingSeq = copy(firingSeq)

                discard = False
                marking_count = copy(potential.getMarkingCount())
                for mark in new_marking:
                    if mark not in marking:
                        if mark in marking_count.keys():
                            marking_count[mark] = marking_count[mark] + 1
                            if marking_count[mark] > self.max_loop:
                                discard = True
                        else:
                            marking_count[mark] = 1

                if enabled_tran.label == None:
                    invs = potential.getInvCounter() + 1
                else:
                    new_firingSeq.append(str(enabled_tran))
                    invs = 0

                if new_marking == self.final_marking:
                    self.generatedTraces.add(tuple(new_firingSeq))
                else:
                    if len(new_firingSeq) < self.maxTraceLength and invs < self.rep_inv_thresh and not discard:
                        self.potentials.put_nowait(PotentialTrace(marking=new_marking, firingSequence=new_firingSeq, inv_counter=invs, marking_count=marking_count))
        return self.generatedTraces


class Sampler():
    def __init__(self, net, initial_marking, final_marking, samples=100):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.samples = samples

    def play(self):
        self.generatedTraces = set()
        for sample in range(self.samples):
            firingSeq = list()
            marking = self.initial_marking
            while marking != self.final_marking:
                enabled_trans = semantics.enabled_transitions(self.net, marking)
                if len(enabled_trans) > 0:
                    tran = random.choice([i for i in enabled_trans])

                    if tran.label != None:
                        firingSeq.append(str(tran))
                    marking = semantics.execute(tran, self.net, marking)
                    self.generatedTraces.add(tuple(firingSeq))
                else:
                    firingSeq = list()
                    marking = self.initial_marking
        return self.generatedTraces
