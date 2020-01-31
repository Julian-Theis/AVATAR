# gan_evaluation

# gan_evaluation_all

# gan_evaluation_mh


# Sample from RelGAN all:
def readTraces(f_name):
    traces = list()
    with open(f_name) as file:
        file_contents = file.read()
        file_contents = file_contents.split("\n")
        for row in file_contents:
            trace = []
            for ev in row.split(" "):
                if len(ev) > 0:
                    trace.append(ev)
            if trace not in traces and len(trace) > 0:
                traces.append(trace)
    return traces

def writeToFile(gan, file, lst):
    iw_dict = gan.iw
    with open(file, 'w') as outfile:
        for trace in lst:
            ttl = []
            for index, event in enumerate(trace):
                if str(event) in iw_dict.keys():
                    w = iw_dict[str(event)]
                    ttl.append(str(w))

            for index, event in enumerate(ttl):
                if index == 0:
                    outfile.write(str(event))
                else:
                    outfile.write(" " + str(event))
            outfile.write("\n")
