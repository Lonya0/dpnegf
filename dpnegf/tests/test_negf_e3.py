from dpnegf.entrypoints.run import run
import pytest
import torch
import numpy as np
import os


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)

def test_negf_e3(root_directory):
    INPUT_file =  root_directory +"/dpnegf/tests/data/test_negf/test_negf_e3/negf_7_0.json"
    output =  root_directory +"/dpnegf/tests/data/test_negf/test_negf_e3/out_negf_e3"
    checkfile =  root_directory +'/dpnegf/tests/data/test_negf/test_negf_e3/nnenv.best.pth'
    structure =  root_directory +"/dpnegf/tests/data/test_negf/test_negf_e3/7_0.vasp"

    run(INPUT=INPUT_file,
        init_model=checkfile,
        structure=structure,
        output=output,
        log_level=5,
        log_path=output+"/output.log")

    
    negf_out_path = output+"/results/negf.out.pth"
    assert os.path.exists(negf_out_path), "NEGF calculation output file not found"
    negf_results = torch.load(negf_out_path,weights_only=False)
    trans = negf_results['T_avg']
    assert(abs(trans[int(len(trans)/2)]-0.2751458)<1e-5)  #compare with calculated transmission at efermi


    if os.path.exists(output+"/results"):
        os.system("rm -r "+output+"/results")

