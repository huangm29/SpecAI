# requires.txt
# alibabacloud_alimt20181012==1.1.0

import sys, os
import requests
import json
import re

# from typing import List

from alibabacloud_alimt20181012.client import Client as alimt20181012Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_alimt20181012 import models as alimt_20181012_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient

# This function finds a chemical name out of a calculation request sentence 
# using the LLM deepseek:32b (locally deployed)
# Input could be a string or a list
def find_chemical_name_from_sentence(sentence):
    API_URL = "http://localhost:11434/api/generate"  # Ollama API port

    headers = {
        "Content-Type": "application/json"
    }
    
    # if input is a list, make it a string
    if isinstance(sentence, list):
        sentence = " ".join(sentence)
        
    data = {
        "model": "deepseek-r1:32b",  # your loaded deepseek model
        # inputs for the LLM API
        "prompt": "Find a chemical name from the following sentence, respond using JSON:  '"+sentence+"' ",
        "stream": False,  # close stream output
        "format": {
            "type": "object",
            "properties": {
            "name": {
                "type": "string"
            },
            },
            "required": [
            "name"
            ]
        }
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        resp =  result["response"];
        nameresult = json.loads(resp);
        return (nameresult["name"]).lower();
    else:
        return ""


# check if any Chinese character is in some text
def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))
    
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

# This function read a .xyz file into a multi-line string
def read_xyz(path_to_file):
    with open(path_to_file) as f:
        f.readline()       # skip the first two lines
        f.readline()       
        data=''.join(line for line in f)
    return data

# Key reference
key_ref = {
    "program": ["orca", "psi4", "nwchem", "gaussian", "cp2k"],
    "method": ["semi-empirical", "am1", "pm3", "pm6", "cndo", "mndo", "nddo", "indo", "zindo", "hf", \
               "dft", "lda", "bp", "blyp", "b3lyp", "pbe", "m06l", "tpss", "scan", "r2scan", "x3lyp", "bhandhlyp", \
               "tpssh", "r2scanh", "wb97x", "camb3lyp", "lc_pbe", "lc_blyp", "wr2scan",
              "mp2", \
              "cis", "cis(d)", "cisd", \
              "cc2", "ccsd", "ccsd(t)"],
    "basis": ["3-21g", "4-31g", "6-31g(d)", "6-31g*", "6-31++g(d,p)", "6-31++g**", "6-311g*", "6-311g(d)", \
              "6-311g**", "6-311g(d,p)", "6-311++g**", "6-311++g(d,p)", "aug-cc-pvtz", "aug-cc-pvdz", \
              "cc-pvdz", "cc-pvtz", "def2-ecp", "def2-svp", "def2-tzvp", "iglo-iii", "lanl2dz", "lanl2tz", \
              "mini", "sapporo-dkh3-dzp-2012", "sapporo-dkh3-tzp-2012", "sto-3g", "stuttgart-rsc-1997"],
    "property": ["single-point", "geometry", "ir", "infrared", "raman", "uv-vis", "uv-visible", 
                 "ultraviolet-visible", "x-ray", \
                 "xas", "xes", "rixs", "xps", "auger", "epr", "esr", "nmr", "mossbauer", "vcd", "roa", "ecd", \
                 "xcd", "photoelectron"],
    "spec_type": ["absorption", "emission"],
    "xray_edge": ["c1s", "o1s", "n1s"]
}

def find_key(sentence, calc_plan, key_ref):
    for word in sentence:
        if word in key_ref["program"]:
            calc_plan.update({'program' : word})
        if word in key_ref["method"]:
            calc_plan.update({'method' : word})
        if word in key_ref["basis"]:
            calc_plan.update({'basis' : word}) 
        if word in key_ref["property"]:
            calc_plan.update({'property' : word}) 
        if word in key_ref["spec_type"]:
            calc_plan.update({'spec_type' : word})
        if word in key_ref["xray_edge"]:
            calc_plan.update({'xray_edge' : word})
            
    return calc_plan

# Create a client of the Alitranslator
def create_client() -> alimt20181012Client:
    """
    @return: Client
    @throws Exception
    """
    config = open_api_models.Config(
    )
    config.endpoint = f'mt.cn-hangzhou.aliyuncs.com'
    return alimt20181012Client(config)

# Ali translator: from Chinese to English
def ali_translator(text):
    client = create_client()
    translate_general_request = alimt_20181012_models.TranslateGeneralRequest(
        format_type='text',
        source_language='zh',
        target_language='en',
        source_text= text,
        scene='general'
    )
    runtime = util_models.RuntimeOptions()
    try:
        response = client.translate_general_with_options(translate_general_request, runtime)
        code = response.body.code
        if code!='200':
            print(response.body.data.message)
            return ''
        else:
            translated_text = response.body.data.translated
            return(translated_text)

    except Exception as error:
        print(error.message)
        print(error.data.get("Recommend"))
        UtilClient.assert_as_string(error.message)   

# If the request is in Chinese, translate it into English
def req2eng_ali(txt):
    if contains_chinese(txt):
        return (ali_translator(txt)).lower()
    else:
        return txt.lower()

# Dictionary for subscript -> normal conversion
# Sometimes LLMs give chemical names with numbers as subscripts
# In order to do pattern matching correctly we need to remove this formatting
def sub_to_normal(str_in):
    dict_ref = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7',
                '₈': '8', '₉': '9'}
    str_out = str_in
    for c in str_in:
        if c in dict_ref:
            str_out = str_out.replace(c, dict_ref[c])
    return str_out
    
# This is a calculation request from user in Chinese
# calc_req = "用ORCA及pbe/3-21g方法计算咖啡因的紫外-可见光谱。"
# calc_req = "计算二氧化碳的红外光谱。"
# calc_req = "计算氯仿的拉曼光谱。"
# calc_req = "计算丁醇溶剂中的咖啡因的紫外可见吸收光谱"
# calc_req = "计算水溶剂中的咖啡因的紫外可见吸收光谱"
# calc_req = "计算H2O溶剂中的咖啡因的紫外可见吸收光谱"
# calc_req = "高通量计算水溶液中的咖啡因的红外光谱"
# calc_req = "高通量计算乙醇溶液中的咖啡因的红外光谱"
# calc_req = "计算丙醇溶液中的Na2CO3的红外光谱" # ionic compound coordinates never be correct!
# calc_req = "计算丙醇溶液中的H2CO3的红外光谱"
# calc_req = "计算水溶液中的分子的红外光谱" # molecular structure not generated!
# calc_req = "计算水溶液中的丁醇的红外光谱"
calc_req = "高通量计算丙酮溶液中的分子的紫外可见吸收光谱"

xyz_file = "mol.xyz"

# First translate the calculation request into English
calc_req_en = req2eng_ali(calc_req)

# Then split the sentence into a word array
delimiters = ",;./ "
calc_req_en = split_sentence(calc_req_en, delimiters)

# cut the solvent description if exists
if ("in" in calc_req_en) and (("solvent" in calc_req_en) or ("solvents" in calc_req_en) or 
                              ("solution" in calc_req_en) or ("solutions" in calc_req_en)):
    in_idx = calc_req_en.index("in")
    sol_desc = calc_req_en[in_idx:] # The solvent description part
# replace 'aqueous' with 'water'. A fix  for inaccurate translations
    sol_desc = [w.replace('aqueous', 'water') for w in sol_desc]

    calc_req_en = calc_req_en[:in_idx] # The other part
    solv_name = (find_chemical_name_from_sentence(sol_desc)).lower()
    solv_name = sub_to_normal(solv_name)
    if solv_name == "":
        solv_name = 'water'
else:
     solv_name = ''   

# Find keys in the calculation request
calc_plan = {}
find_key(calc_req_en, calc_plan, key_ref)

# Default values
if "program" not in calc_plan:
    calc_plan.update({"program" : "orca"})

if "method" not in calc_plan:
    calc_plan.update({"method" : "b3lyp"})
        
if "basis" not in calc_plan:
    calc_plan.update({"basis" : "def2-svp"})
        
if "property" not in calc_plan:
    calc_plan.update({"property" : "single-point"})
                     
if "geom" not in calc_plan:
    calc_plan.update({"geom" : {"type" : "xyz", "unit" : "angstrom"}})
                     
if "charge" not in calc_plan:
    calc_plan.update({"charge" : "0"})
                     
if "spin" not in calc_plan:
    calc_plan.update({"spin" : "1"})
                     
# Alternative values
if calc_plan["method"] == "dft":
    calc_plan["method"] = "b3lyp"
    
if calc_plan["basis"] == "6-31g(d)":
    calc_plan["basis"] = "6-31g*"
    
if calc_plan["basis"] == "6-31++g(d,p)":
    calc_plan["basis"] = "6-31++g**"
    
if calc_plan["basis"] == "6-311g(d)":
    calc_plan["basis"] = "6-311g*"
    
if calc_plan["basis"] == "6-311g(d,p)":
    calc_plan["basis"] = "6-311g**"
    
if calc_plan["basis"] == "6-311++g(d,p)":
    calc_plan["basis"] = "6-311++g**"
    
if calc_plan["property"] == "uv-visible":
    calc_plan["property"] = "uv-vis"
    
if calc_plan["property"] == "ultraviolet-visible":
    calc_plan["property"] = "uv-vis"

if calc_plan["property"] in ["uv-vis", "ecd", "xas", "xcd"]:
    calc_plan.update({"n_ex_states" : "30"}) # default to calculate 30 excited states
    
if calc_plan["property"] == "infrared":
    calc_plan["property"] = "ir"
    
if calc_plan["property"] == "ir":
    calc_plan.update({"calc_type" : "opt freq"})
    
if calc_plan["property"] == "raman":
    calc_plan.update({"calc_type" : "opt numfreq"})

if solv_name != "":
    calc_plan.update({"solvent" : solv_name})

# The default spectroscopy type is absorption
if (calc_plan["property"] in ["uv-vis", "x-ray", "xas"]) and ("spec_type" not in calc_plan):
    calc_plan.update({"spec_type" : "absorption"})    

# read the ORCA supported solvent names in
if 'solvent' in calc_plan:
    orca_solvents = ['1,1,1-trichloroethane', '1,1,2-trichloroethane', '1,2,4-trimethylbenzene',
                     '1,2-dibromoethane', '1,2-dichloroethane', '1,2-ethanediol', '1,4-dioxane',
                     'dioxane', '1-bromo-2-methylpropane', '1-bromooctane', 'bromooctane', '1-bromopentane',
                     '1-bromopropane', '1-butanol', 'butanol', '1-chlorohexane', 'chlorohexane',
                     '1-chloropentane', '1-chloropropane', '1-decanol', 'decanol', '1-fluorooctane',
                     '1-heptanol', 'heptanol', '1-hexanol', 'hexanol', '1-hexene', '1-hexyne', '1-iodobutane',
                     '1-iodohexadecane', 'hexadecyliodide', '1-iodopentane', '1-iodopropane', '1-nitropropane',
                     '1-nonanol', 'nonanol', '1-octanol', 'octanol', '1-pentanol', 'pentanol', '1-pentene',
                     '1-propanol', 'propanol', '2,2,2-trifluoroethanol', '2,2,4-trimethylpentane', 'isooctane',
                     '2,4-dimethylpentane', '2,4-dimethylpyridine', '2,6-dimethylpyridine', '2-bromopropane',
                     '2-butanol', 'secbutanol', '2-chlorobutane', '2-heptanone', '2-hexanone',
                     '2-methoxyethanol', 'methoxyethanol', '2-methyl-1-propanol', 'isobutanol',
                     '2-methyl-2-propanol', '2-methylpentane', '2-methylpyridine', '2methylpyridine',
                     '2-nitropropane', '2-octanone', '2-pentanone', '2-propanol', 'isopropanol',
                     '2-propen-1-ol', 'e-2-pentene', '3-methylpyridine', '3-pentanone', '4-heptanone',
                     '4-methyl-2-pentanone', '4methyl2pentanone', '4-methylpyridine', '5-nonanone', 
                     'acetic acid', 'aceticacid', 'acetone', 'acetonitrile', 'mecn', 'ch3cn', 'acetophenone',
                     'ammonia', 'aniline', 'anisole', 'benzaldehyde', 'benzene', 'benzonitrile', 
                     'benzyl alcohol', 'benzylalcohol', 'bromobenzene', 'bromoethane', 'bromoform', 'butanal',
                     'butanoic acid', 'butanone', 'butanonitrile', 'butyl ethanoate', 'butyl acetate',
                     'butylacetate', 'butylamine', 'n-butylbenzene', 'butylbenzene', 'sec-butylbenzene',
                     'secbutylbenzene', 'tert-butylbenzene', 'tbutylbenzene', 'carbon disulfide',
                     'carbondisulfide', 'cs2', 'carbon tetrachloride', 'ccl4', 'chlorobenzene', 'chloroform',
                     'chcl3', 'a-chlorotoluene', 'o-chlorotoluene', 'conductor', 'm-cresol', 'mcresol',
                     'o-cresol', 'cyclohexane', 'cyclohexanone', 'cyclopentane', 'cyclopentanol',
                     'cyclopentanone', 'decalin', 'cis-decalin', 'n-decane', 'decane', 'dibromomethane',
                     'dibutylether', 'o-dichlorobenzene', 'odichlorobenzene', 'e-1,2-dichloroethene',
                     'z-1,2-dichloroethene', 'dichloromethane', 'ch2cl2', 'dcm', 'diethyl ether',
                     'diethylether', 'diethyl sulfide', 'diethylamine', 'diiodomethane', 'diisopropyl ether',
                     'diisopropylether', 'cis-1,2-dimethylcyclohexane', 'dimethyl disulfide',
                     'n,n-dimethylacetamide', 'dimethylacetamide', 'n,n-dimethylformamide', 'dimethylformamide',
                     'dmf', 'dimethylsulfoxide', 'dmso', 'diphenylether', 'dipropylamine', 'n-dodecane',
                     'dodecane', 'ethanethiol', 'ethanol', 'ethyl acetate', 'ethylacetate', 'ethanoate',
                     'ethyl methanoate','ethyl phenyl ether', 'ethoxybenzene', 'ethylbenzene', 'fluorobenzene',
                     'formamide', 'formic acid', 'furan', 'furane', 'n-heptane', 'heptane', 'n-hexadecane',
                     'hexadecane', 'n-hexane', 'hexane', 'hexanoic acid', 'iodobenzene', 'iodoethane',
                     'iodomethane', 'isopropylbenzene', 'p-isopropyltoluene', 'isopropyltoluene', 'mesitylene',
                     'methanol', 'methyl benzoate', 'methyl butanoate', 'methyl ethanoate', 'methyl methanoate',
                     'methyl propanoate', 'n-methylaniline', 'methylcyclohexane', 'n-methylformamide',
                     'methylformamide', 'nitrobenzene', 'phno2', 'nitroethane', 'nitromethane', 'meno2',
                     'o-nitrotoluene', 'onitrotoluene', 'n-nonane', 'nonane', 'n-octane', 'octane',
                     'n-pentadecane', 'pentadecane', 'octanol(wet)', 'wetoctanol', 'woctanol', 'pentanal',
                     'n-pentane', 'pentane', 'pentanoic acid', 'pentyl ethanoate', 'pentylamine',
                     'perfluorobenzene', 'hexafluorobenzene', 'phenol', 'propanal', 'propanoic acid',
                     'propanonitrile', 'propyl ethanoate', 'propylamine', 'pyridine', 'tetrachloroethene',
                     'c2cl4', 'tetrahydrofuran', 'thf', 'tetrahydrothiophene-s,s-dioxide',
                     'tetrahydrothiophenedioxide', 'sulfolane', 'tetralin', 'thiophene', 'thiophenol',
                     'toluene', 'trans-decalin', 'tributylphosphate', 'trichloroethene', 'triethylamine',
                     'n-undecane', 'undecane', 'water', 'h2o', 'xylene', 'm-xylene', 'o-xylene', 'p-xylene']
    
    if calc_plan['solvent'] not in orca_solvents:
        print("Solvent \""+solv_name+"\" not recognized or not supported!")
        sys.exit()
    
def Input_gen_orca_xyz(plan, XYZ_FILE):

# Create the method section of the ORCA calculation
    if "calc_type" in plan:
        tmp_str1 = '!'+plan["method"]+' '+plan['basis']+' '+plan['calc_type']+'\n' # the method line
    else:
        tmp_str1 = '!'+plan["method"]+' '+plan['basis']+'\n'
    if "solvent" in plan:
        tmp_str1 = tmp_str1[:-1] + " CPCM(" + plan["solvent"] + ")\n"
# TDDFT section
    if 'n_ex_states' in plan:
        tmp_str2 = '\n%TDDFT\n   NROOTS   '+plan['n_ex_states']+'\nEND\n'
    else:
        tmp_str2 =''
        
# elprop section
    if plan["property"] == "raman":
        tmp_str2 = '\n%elprop\nPolar  1\nend\n'

# molecular geom section
    tmp_str3 = '\n*'+calc_plan["geom"]["type"]+' '+calc_plan["charge"]+' '+calc_plan["spin"]+'\n'+read_xyz(XYZ_FILE)+'*'
    
    return tmp_str1 + tmp_str2 + tmp_str3 # return the input file as a multiple line string
                     
                                              
if calc_plan["program"] == "orca":
    input_str = Input_gen_orca_xyz(calc_plan, xyz_file)
#     print("\nThe prepared " + calc_plan["program"] + " input file is:\n")
    print(input_str)
else:
    print("Program input generation not implemented yet!")
    sys.exit()
