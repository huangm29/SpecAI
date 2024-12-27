# This script read the outputs from ga_print() or ga_print_file()
# and store the GAs as python arrays

import numpy as np
import sys

def read_ga_print(filename):
    # locate the head of the printout of a GA and parse for the name
    M_list = [] # list of matrices read from the file
    with open(filename,"r") as IN:
        for line in IN:
            if line.startswith(" global array:"):
                line_sp = ((",".join((",".join(line.split("["))).split(":"))).replace("]","")).split(",")
                name = line_sp[1].strip()
                r_dim = int(line_sp[3])
                c_dim = int(line_sp[5])
                
                #print(line_sp)
                #print(r_dim, c_dim)
                #sys.exit()

                Mat = np.zeros((r_dim, c_dim))

                # read m = int(c_dim/6) + 1 blocks of data
                for i in range(int(c_dim/6) + 1):
                    # read n = 3 + r_dim lines in one block of data
                    for j in range(3 + r_dim):
                        line1 = IN.readline()
                        # skip first 3 lines
                        if j > 2:
                            line1_sp = line1.split()
                            for k in range(len(line1_sp) - 1):
                                Mat[j-3, i*6 + k] = float(line1_sp[k+1])

                # finish reading one matrix, append it to the list
                M_list.append([name,Mat])

    return M_list


INPUT_FILE = "ga_print-test.dat"

test_M_list = read_ga_print(INPUT_FILE)

#print(test_M_list[0][0])
#print(test_M_list[0][1].shape)
print(test_M_list[1][1][10,16])
#print(test_M_list[1][0])
#print(test_M_list[1][1].shape)
#print(test_M_list[2][0])
#print(test_M_list[2][1].shape)
