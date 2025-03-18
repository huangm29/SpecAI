import re
import sys, os
import requests
import json

# This function cuts the IR or Raman section from the output
def cut_IR_RAMAN(path_to_file, spec_type = 'ir'):
    with open(path_to_file, "r") as IN:
        mstr = ''''''
        for line in IN:
            line_sp = line.split()
            if len(line_sp) != 0:
                if spec_type.lower() == 'ir':
                    if line_sp[0] == 'IR' and line_sp[1] == 'SPECTRUM':
                        while (line != '\n'):
                            line = IN.readline()
                        while line == '\n':
                            line = IN.readline()
                        while line != '\n':
                            mstr += line
                            line = IN.readline()
                elif spec_type.lower() == 'raman':
                    if line_sp[0] == 'RAMAN' and line_sp[1] == 'SPECTRUM':
                        while (line != '\n'):
                            line = IN.readline()
                        while line == '\n':
                            line = IN.readline()
                        while line != '\n':
                            mstr += line
                            line = IN.readline()
                else:
                    print("Spectral type not recoginized in cut_IR_RAMAN()!")
    if len(mstr) == 0:
        print("Spectral data not found!")
    return mstr

# This function cuts the absorption or CD sections from the output
def cut_AS_CD(path_to_file, gauge = 'electric', spec_type = 'absorption', 
        high_moment = False, origin_indep = False, semi_classical = False):
    with open(path_to_file, "r") as IN:
        mstr = ''''''
        for line in IN:
            if len(line) != 0:
                if semi_classical:
                    if spec_type.lower() == 'absorption':
                        if 'ABSORPTION SPECTRUM VIA FULL SEMI-CLASSICAL' in line:
                            line = IN.readline()
                            while line != '\n':
                                mstr += line
                                line = IN.readline()
                    elif spec_type.lower() == 'cd':
                            if 'CD SPECTRUM VIA FULL SEMI-CLASSICAL' in line:
                                line = IN.readline()
                                while line != '\n':
                                    mstr += line
                                    line = IN.readline()
                else:
                    if gauge.lower() == 'electric':
                        if spec_type == 'absorption':
                            if not high_moment:
                                if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC' in line:
                                    line = IN.readline()
                                    while line != '\n':
                                        mstr += line
                                        line = IN.readline()
                            elif ('ABSORPTION SPECTRUM COMBINED') in line and (not 'Velocity' in line):
                                    line = IN.readline()
                                    while line != '\n':
                                        mstr += line
                                        line = IN.readline()
                        elif spec_type.lower() == 'cd':
                            if 'CD SPECTRUM VIA TRANSITION ELECTRIC' in line:
                                line = IN.readline()
                                while line != '\n':
                                    mstr += line
                                    line = IN.readline()
                    elif gauge.lower() == 'velocity':
                        if spec_type.lower() == 'absorption':
                            if not high_moment:
                                if 'ABSORPTION SPECTRUM VIA TRANSITION VELOCITY' in line:
                                    line = IN.readline()
                                    while line != '\n':
                                        mstr += line
                                        line = IN.readline()
                            elif origin_indep:
                                if ('ABSORPTION SPECTRUM COMBINED') in line and ('Velocity' in line) and ( 'Origin Independent' in line):
                                    line = IN.readline()
                                    while line != '\n':
                                        mstr += line
                                        line = IN.readline()
                            else:
                                if ('ABSORPTION SPECTRUM COMBINED') in line and ('Velocity' in line) and ( not 'Origin Independent' in line):
                                    line = IN.readline()
                                    while line != '\n':
                                        mstr += line
                                        line = IN.readline()
                        elif spec_type.lower() == 'cd':
                            if 'CD SPECTRUM VIA TRANSITION VELOCITY' in line:
                                line = IN.readline()
                                while line != '\n':
                                    mstr += line
                                    line = IN.readline()
                    else:
                        print("Gauge not recognized in cut_AS_CD()!")
    if len(mstr) == 0:
        print("Spectral data not found!")
    return mstr

def extract_tabular_data(content, column1, column2):
    API_URL = "http://localhost:11434/api/generate"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-r1:32b",  
        "prompt": """Suppose we have the following tabular data as the following:

        """+content+" , please list all data in the \'"+column1+"\' and \'"+column2+"\' columns, respond in JSON",
        "options": {
            "temperature": 0
        },  
        "stream": False 

    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        resp =  result["response"]
        print(resp)
        think_end_length = 8
        think_end = resp.find("</think>")

        regular_result = resp
        if think_end>=0 :
            regular_result = resp[think_end+think_end_length:]
        json_start = regular_result.find("```json")
        json_start_length = 7
        json_resut = regular_result
        if json_start>=0 :
            json_resut = regular_result[json_start+json_start_length:]
        json_end = json_resut.find("```")
        if json_end>=0 :
            json_resut = json_resut[0:json_end]
        return json.loads(json_resut)
    else:
        return {}
    
    try:
        tables = json.loads(llm_result)
        return tables
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON from LLM result: {e}"}

def find_nth_occur(string, substr, n):
    if n == 0:
        return 0
    elif n < 0:
        print("Wrong occurrence number!")
        sys.exit()
    else:
        idx_list = [m.start() for m in re.finditer(substr, string)]
        if n > len(idx_list):
            print("There are no "+n+" occurences in the string!")
            sys.exit()
        else:
            return [m.start() for m in re.finditer(substr, string)][n - 1]
        
def last_line(string):
    idx_list = [m.start() for m in re.finditer('\n', string)]
    len_idx_list = len(idx_list)
    if len_idx_list == 0:
        return string
    elif string[-1] == '\n':
        return string[idx_list[-2]:-1]
    else:
        return string[idx_list[-1]:]

def first_n_lines(string, n):
    idx = find_nth_occur(tab_data, '\n', n)
    return string[:idx]

def split_sentence(sentence, delimiters):
  """
  Splits a sentence into a list of words using multiple delimiters.

  Args:
    sentence: The sentence to split.
    delimiters: A string containing the delimiters to split by, 
                e.g., ",| ".

  Returns:
    A list of words.
  """
  regex_pattern = '[' + re.escape(delimiters) + ']+'
  words = re.split(regex_pattern, sentence)
  return [word.lower() for word in words if word]

def smart_extract_tab(tab_data, col1, col2):
    tab_f8 = first_n_lines(tab_data, 8)

    tab_dict = extract_tabular_data(tab_f8, col1, col2)
    #print(tab_dict)
    if "data" in tab_dict:
        tab_dict = tab_dict["data"]
    for key in tab_dict[0]:
        delimiters = "()_"
        if (split_sentence(key, delimiters)[0]).lower() == (split_sentence(col1, delimiters)[0]).lower():
            col1 = key
            continue
        if (split_sentence(key, delimiters)[0]).lower() == (split_sentence(col2, delimiters)[0]).lower():
            col2 = key
            continue
    
    l_line = last_line(tab_f8)
    
    l_line_sp = l_line.split()
    len_line = len(l_line) - 1
    idx_col1 = l_line_sp.index(str(tab_dict[-1][col1]))
    idx_col2 = l_line_sp.index(str(tab_dict[-1][col2]))

    line8_idx = find_nth_occur(tab_data, '\n', 8)

    tab_data_tail = tab_data[line8_idx + 1:]

    line_idx_list = [m.start() for m in re.finditer('\n', tab_data_tail)]

    for i in range(len(line_idx_list)):
        if i == 0:
            line = tab_data_tail[:line_idx_list[0]]
        else:
            line = tab_data_tail[line_idx_list[i - 1] + 1: line_idx_list[i]]
        if len(line) == len_line:
            line_sp = line.split()
            tab_dict.append({col1 : float(line_sp[idx_col1]), col2 : float(line_sp[idx_col2])})
    
    if tab_data[-1] != "\n":
        line = tab_data_tail[line_idx_list[i]:]
        if len(line) == len_line:
            line_sp = line.split()
            tab_dict.append({col1 : float(line_sp[idx_col1]), col2 : float(line_sp[idx_col2])})
        
    col1_data = []
    col2_data = []
    for row in tab_dict:
        col1_data.append(row[col1])
        col2_data.append(row[col2])
    
    return col1_data, col2_data

tab_data = '''                     ABSORPTION SPECTRUM COMBINED ELECTRIC DIPOLE + MAGNETIC DIPOLE + ELECTRIC QUADRUPOLE SPECTRUM (Origin Independent, Velocity)
---------------------------------------------------------------------------------------------------------------------------------
     Transition      Energy     Energy  Wavelength fosc(P2)  fosc(M2)  fosc(Q2) fosc(P2+M2+Q2+PM+PO) P2/TOT    M2/TOT    Q2/TOT
                      (eV)      (cm-1)    (nm)       (au)    (au*1e6)  (au*1e6)
---------------------------------------------------------------------------------------------------------------------------------
  0-1A  ->  1-1A   520.666069 4199455.5     2.4     0.01891 685.44588 410.85376   0.01889742982163   1.00056   0.03627   0.02174
  0-1A  ->  2-1A   522.468872 4213996.1     2.4     0.00296  40.68873  78.25483   0.00295539966464   1.00075   0.01377   0.02648
  0-1A  ->  3-1A   522.730656 4216107.5     2.4     0.01078  18.39742 313.58140   0.01076904723339   1.00080   0.00171   0.02912
  0-1A  ->  4-1A   523.021838 4218456.1     2.4     0.00293  91.57893  67.89888   0.00293200407039   1.00083   0.03123   0.02316
  0-1A  ->  5-1A   523.502292 4222331.2     2.4     0.00615 131.57085 154.92564   0.00614503354222   1.00075   0.02141   0.02521
  0-1A  ->  6-1A   524.190116 4227878.9     2.4     0.00031  11.33675   7.02799   0.00030986415686   1.00067   0.03659   0.02268
  0-1A  ->  7-1A   524.338199 4229073.2     2.4     0.00497 100.88548 126.78344   0.00496312860016   1.00077   0.02033   0.02555
  0-1A  ->  8-1A   524.489473 4230293.3     2.4     0.00056  17.03448  12.94918   0.00055701656019   1.00056   0.03058   0.02325
  0-1A  ->  9-1A   524.611908 4231280.8     2.4     0.00043  10.87472  10.49293   0.00042819926408   1.00066   0.02540   0.02450
  0-1A  -> 10-1A   524.978817 4234240.2     2.4     0.00266  34.60695  71.89320   0.00265393629023   1.00082   0.01304   0.02709
  0-1A  -> 11-1A   525.473033 4238226.3     2.4     0.01446 150.76402 401.84876   0.01444515130930   1.00079   0.01044   0.02782
  0-1A  -> 12-1A   526.107865 4243346.5     2.4     0.00163  56.17113  37.46690   0.00162423523401   1.00058   0.03458   0.02307
  0-1A  -> 13-1A   526.276229 4244704.5     2.4     0.00239  87.18469  54.24188   0.00239249560924   1.00079   0.03644   0.02267
  0-1A  -> 14-1A   526.536875 4246806.7     2.4     0.00127  47.32173  28.73303   0.00127258180234   1.00052   0.03719   0.02258
  0-1A  -> 15-1A   532.949577 4298528.7     2.3     0.00060   5.41564  17.66150   0.00059737061802   1.00081   0.00907   0.02957
  0-1A  -> 16-1A   533.137466 4300044.1     2.3     0.00678 155.90738 180.01280   0.00677144809946   1.00062   0.02302   0.02658
  0-1A  -> 17-1A   534.081867 4307661.2     2.3     0.00417  21.99343 127.02369   0.00417129848235   1.00060   0.00527   0.03045
  0-1A  -> 18-1A   534.634341 4312117.2     2.3     0.01370 367.15497 349.26813   0.01368671567416   1.00071   0.02683   0.02552
  0-1A  -> 19-1A   535.177935 4316501.6     2.3     0.00147  56.24137  32.45658   0.00146450209030   1.00051   0.03840   0.02216
  0-1A  -> 20-1A   535.831064 4321769.4     2.3     0.00193  71.86327  46.16281   0.00192771082418   1.00053   0.03728   0.02395
  0-1A  -> 21-1A   536.786099 4329472.3     2.3     0.00322 117.01429  77.32639   0.00322168603218   1.00052   0.03632   0.02400
  0-1A  -> 22-1A   537.147644 4332388.4     2.3     0.00082  31.56218  18.67964   0.00081479457994   1.00036   0.03874   0.02293
  0-1A  -> 23-1A   537.850856 4338060.2     2.3     0.00929 238.41697 240.62040   0.00928207793692   1.00044   0.02569   0.02592
  0-1A  -> 24-1A   539.564556 4351882.1     2.3     0.01118 288.91983 285.82404   0.01117947332142   1.00047   0.02584   0.02557
  0-1A  -> 25-1A   540.344260 4358170.8     2.3     0.02315 600.96953 599.95263   0.02314052968984   1.00052   0.02597   0.02593
  0-1A  -> 26-1A   540.613163 4360339.7     2.3     0.03062 507.26131 849.92930   0.03060407493974   1.00045   0.01657   0.02777
  0-1A  -> 27-1A   542.473617 4375345.2     2.3     0.00561 123.38267 151.64869   0.00560777322780   1.00069   0.02200   0.02704
  0-1A  -> 28-1A   542.709978 4377251.6     2.3     0.00255  94.59974  61.89760   0.00255130520467   1.00043   0.03708   0.02426
  0-1A  -> 29-1A   543.635728 4384718.3     2.3     0.00640 153.17752 170.35107   0.00639402369329   1.00062   0.02396   0.02664
  0-1A  -> 30-1A   544.031382 4387909.5     2.3     0.00245  54.11068  66.33958   0.00245196425218   1.00058   0.02207   0.02706
*The oscillator strengths printed in this table have been obtained from a formalism that makes them INDEPENDENT of the choice of origin.
Please refer to the manual for further information.
'''

col1 = 'Energy(eV)'
col2 = 'fosc(P2+M2+Q2+PM+PO)'

energy, fsoc = smart_extract_tab(tab_data, col1, col2)

for i in range(len(energy)):
    print(energy[i], fsoc[i])
