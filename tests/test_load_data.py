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


def test_wrong_file_path():
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_data("dane AG 2/large_scale/knapPI_2_200_100d")


def test_empty_file(tmp_path):
    with pytest.raises(ValueError, match="File is empty or corrupted"):
        bad_data_file = tmp_path / "corrupted_data_file"
        bad_data_file.write_text(" ")
        load_data(bad_data_file)


def test_corrupted_file(tmp_path):
    bad_data_file = tmp_path / "corrupted_data_file"
    bad_data_file.write_text("\n30\n3\n4 4\n 9 0 9\n \n \n")
    try:
        result = load_data(bad_data_file)
        print("\n--- DEBUG: load_data returned ---")
        print(result)
        print("--------------------------------")
        pytest.fail("Nie został rzucony ValueError dla uszkodzonego pliku")
    except ValueError as e:
        assert "File is empty or corrupted" in str(e)
