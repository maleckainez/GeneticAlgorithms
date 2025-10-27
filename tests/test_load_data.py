import pytest
from src.methods.utils import load_data


def test_correct_small_file():
    data = load_data("../dane AG 2/low-dimensional/f1_l-d_kp_10_269")
    assert isinstance(data, dict)
    assert len(data) > 0
    first_key = next(iter(data))
    assert len(data[first_key]) == 2


def test_correct_large_file():
    data = load_data("../dane AG 2/large_scale/knapPI_2_200_1000_1")
    assert isinstance(data, dict)
    assert len(data) > 0
    first_key = next(iter(data))
    assert len(data[first_key]) == 2


def test_correct_file_with_multiple_empty_spaces(tmp_path):
    correct_data = tmp_path / "correct_file_data"
    correct_data.write_text("1 0 \n \n \n \n 90 9 \n 9 0\n 5 6\n")
    correct_result = {0: [1, 0], 1: [90, 9], 2: [9, 0], 3: [5, 6]}
    try:
        data = load_data(correct_data)
        assert correct_result == data

    except Exception:
        raise Exception


def test_wrong_file_path():
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_data("dane AG 2/large_scale/knapPI_2_200_100d")


def test_empty_file(tmp_path):
    with pytest.raises(ValueError, match="File is empty"):
        bad_data_file = tmp_path / "corrupted_data_file"
        bad_data_file.write_text(" ")
        load_data(bad_data_file)


def test_too_many_num_in_data(tmp_path):
    bad_data_file = tmp_path / "corrupted_data_file"
    bad_data_file.write_text("\n30 3\n4 4\n 9 0 9\n \n \n")
    with pytest.raises(
        ValueError, match="Invalid values on line 3: expected 2 values, got 3"
    ):
        load_data(bad_data_file)


def test_letter_in_data(tmp_path):
    bad_data_file = tmp_path / "corrupted_data_file"
    bad_data_file.write_text("\n30 a\n4 4\n 9 0\n \n \n \n")
    with pytest.raises(
        ValueError, match="Invalid values on line 1: received non integer input 30 a"
    ):
        load_data(bad_data_file)


def test_not_not_enough_num_in_data(tmp_path):
    bad_data_file = tmp_path / "corrupted_data_file"
    bad_data_file.write_text("\n30 4\n4 4\n 9 ")
    with pytest.raises(
        ValueError, match="Invalid values on line 3: expected 2 values, got 1"
    ):
        load_data(bad_data_file)


def test_non_int_data(tmp_path):
    bad_data_file = tmp_path / "float_data_file"
    bad_data_file.write_text("\n30.1 4.0\n4.1 4.3\n 9.7 8.9 ")
    with pytest.raises(
        ValueError,
        match="Invalid values on line 1: received non integer input 30.1 4.0",
    ):
        load_data(bad_data_file)
