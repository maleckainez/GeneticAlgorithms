from src.methods.utils import load_data
from pathlib import Path
import pytest


def test_loading_included_low_dimension_files(root_path):
    low_dimension_directory = Path(root_path / "dane AG 2" / "low-dimensional")
    file1 = Path(low_dimension_directory / "f6_l-d_kp_10_60")
    file2 = Path(low_dimension_directory / "f7_l-d_kp_7_50")
    try:
        file1_data = load_data(file1)
        file2_data = load_data(file2)
    except Exception as e:
        raise e
    assert file1_data.shape == (11, 2)
    assert file2_data.shape == (8, 2)
    assert (file1_data[3, 0] == 17) and (file1_data[3, 1] == 20)
    assert (file2_data[1, 0] == 70) and (file2_data[1, 1] == 31)
    assert sum(file1_data[:, 0]) == 115
    assert sum(file2_data[:, 1]) == 143


def test_loading_included_low_dimension_files(root_path):
    high_dimension_directory = Path(root_path / "dane AG 2" / "large_scale")
    file1 = Path(high_dimension_directory / "knapPI_1_100_1000_1")
    file2 = Path(high_dimension_directory / "knapPI_2_1000_1000_1")
    try:
        file1_data = load_data(file1)
        file2_data = load_data(file2)
    except Exception as e:
        raise e
    assert file1_data.shape == (101, 2)
    assert file2_data.shape == (1001, 2)
    assert (file1_data[74, 0] == 65) and (file1_data[74, 1] == 585)
    assert (file2_data[453, 0] == 271) and (file2_data[453, 1] == 190)
    assert sum(file1_data[:, 0]) == 50144
    assert sum(file2_data[:, 1]) == 510292


def test_loading_non_existing_file(temp_file):
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_data(temp_file)


def test_loading_file_containing_non_integer_values(temp_file):
    temp_file.write_text("\n \n \n 30 3\n 49  33\n 3.6      34.5\n \n")
    with pytest.raises(
        ValueError,
        match="Invalid values on line 3: received non integer input 3.6      34.5",
    ):
        load_data(temp_file)


def test_loading_empty_file(temp_file):
    temp_file.write_text(" ")
    with pytest.raises(ValueError, match="File is empty"):
        load_data(temp_file)


def test_loading_too_many_num_in_line(temp_file):
    temp_file.write_text("\n30 3\n4 4\n 9 0 9\n \n \n")
    with pytest.raises(
        ValueError, match="Invalid values on line 3: expected 2 values, got 3"
    ):
        load_data(temp_file)


def test_loading_letter_in_data(temp_file):
    temp_file.write_text("\n30 a\n4 4\n 9 0\n \n \n \n")
    with pytest.raises(
        ValueError, match="Invalid values on line 1: received non integer input 30 a"
    ):
        load_data(temp_file)


def test_loading_not_not_enough_num_in_line(temp_file):
    temp_file.write_text("\n30 4\n4 4\n 9 ")
    with pytest.raises(
        ValueError, match="Invalid values on line 3: expected 2 values, got 1"
    ):
        load_data(temp_file)


def test_loading_correct_file_with_multiple_whitespaces(temp_file):
    import numpy as np

    temp_file.write_text(
        "1 0 \n \n \n \n 90                   9 \n    9   0\n 5    6\n"
    )
    expected_result = np.array([[1, 0], [90, 9], [9, 0], [5, 6]])
    data = load_data(temp_file)
    np.testing.assert_array_equal(data, expected_result)
