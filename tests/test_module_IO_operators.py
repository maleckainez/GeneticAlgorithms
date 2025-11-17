from src.methods.utils import load_data
from pathlib import Path
import pytest

def test_loading_included_low_dimension_files(root_path):
    low_dimension_directory = Path(root_path/"dane AG 2"/"low-dimensional")
    file1 = Path(low_dimension_directory/"f6_l-d_kp_10_60")
    file2 = Path(low_dimension_directory/"f7_l-d_kp_7_50")
    try:
        file1_data = load_data(file1)
        file2_data = load_data(file2)
    except Exception as e:
        raise e
    assert file1_data.shape == (11,2)
    assert file2_data.shape == (8,2) 
    assert (file1_data[3,0] == 17) and (file1_data[3,1] == 20)
    assert (file2_data[1,0] == 70) and (file2_data[1,1] == 31)
    assert sum(file1_data[:,0]) == 115
    assert sum(file2_data[:,1]) == 143

def test_loading_included_low_dimension_files(root_path):
    high_dimension_directory = Path(root_path/"dane AG 2"/"large_scale")
    file1 = Path(high_dimension_directory/"knapPI_1_100_1000_1")
    file2 = Path(high_dimension_directory/"knapPI_2_1000_1000_1")
    try:
        file1_data = load_data(file1)
        file2_data = load_data(file2)
    except Exception as e:
        raise e
    assert file1_data.shape == (101,2)
    assert file2_data.shape == (1001,2) 
    assert (file1_data[74,0] == 65) and (file1_data[74,1] == 585)  
    assert (file2_data[453,0] == 271) and (file2_data[453,1] == 190)  
    assert sum(file1_data[:,0]) == 50144
    assert sum(file2_data[:,1]) == 510292

def test_loading_non_existing_file(root_path):
    non_existing_directory = Path(root_path/"non_existing_filepath")
    file = Path(non_existing_directory/"non_existing_file")
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_data(file)