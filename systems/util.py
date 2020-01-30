def writeToFile(file, lst):
    with open(file, 'w') as outfile:
        for entry in lst:
            print_trace = ""
            for index, ev in enumerate(entry):
                if index == 0:
                    print_trace = str(ev).replace(" ", "")
                else:
                    print_trace = print_trace + " " + str(ev).replace(" ", "")
            outfile.write(print_trace.strip() + "\n")