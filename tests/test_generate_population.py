from src.methods.utils import create_population_file
from pathlib import Path
import json


def _validate_population_file(pr, rng, func_name):
    temp_path = pr.get_temp_path()
    name = pr.filename_constant
    assert temp_path.exists()
    assert "pytest_temp_file" in str(temp_path)
    func_name(temp_path, rng, name)
    mmap_file = Path(temp_path/f"{name}.dat")
    json_file = Path(temp_path/f"{name}.json")
    assert mmap_file.exists()
    assert json_file.exists()
    assert mmap_file.stat().st_size != 0
    assert json_file.stat().st_size != 0
    with open(json_file, "r") as f:
        config = json.load(f)
    filesize = config["filesize"]
    assert filesize == mmap_file.stat().st_size

def _small_pop(temp_path, rng, name):
    create_population_file(
        population_size=10,
        genome_length= 10,
        stream_batch= 500,
        rng= rng,
        temp= temp_path,
        probability_of_failure= None,
        filename_constant= name
    )

def _wide_pop(temp_path, rng, name):
    create_population_file(
        population_size=10,
        genome_length= 10000,
        stream_batch= 1,
        rng= rng,
        temp= temp_path,
        probability_of_failure= None,
        filename_constant= name
    )



def test_create_small_pop(test_only_pathresolver, test_only_rng):
    _validate_population_file(
        pr=test_only_pathresolver, 
        rng=test_only_rng, 
        func_name=_small_pop)
    
def test_create_wide_pop(test_only_pathresolver, test_only_rng):
    _validate_population_file(
        pr=test_only_pathresolver, 
        rng=test_only_rng, 
        func_name=_wide_pop)
