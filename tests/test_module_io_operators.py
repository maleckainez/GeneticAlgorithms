# pylint: disable=missing-function-docstring
"""Defines tests for i/o functions.

This module contiains tests for every function modifying
external files in any way. They should all be located in
``src.methods.data_loader`` and ``src.methods.memmap_operations``.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from src.methods.data_loader import load_data, load_yaml_config
from src.methods.memmap_operations import create_memmap_config_json, load_memmap


def test_loading_included_low_dimensional_files(root_path):
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
    temp_file.write_text(
        "1 0 \n \n \n \n 90                   9 \n    9   0\n 5    6\n"
    )
    expected_result = np.array([[1, 0], [90, 9], [9, 0], [5, 6]])
    data = load_data(temp_file)
    np.testing.assert_array_equal(data, expected_result)


def test_create_json_with_correct_input(temp_file):

    create_memmap_config_json(
        path=temp_file,
        dat_path=temp_file,
        datatype=np.uint8,
        population_size=100,
        genome_length=500,
    )
    assert temp_file.exists()
    with open(temp_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config["filename"] == str(temp_file)
    assert config["data_type"] == np.dtype(np.uint8).name
    assert config["population_size"] == 100
    assert config["genome_length"] == 500
    assert config["filesize"] == 500 * 100 * np.dtype(np.uint8).itemsize


def test_create_json_with_string_dat_path(temp_file):
    create_memmap_config_json(
        path=temp_file,
        dat_path="string_dat_path",
        datatype=np.uint8,
        population_size=100,
        genome_length=500,
    )
    assert temp_file.exists()
    with open(temp_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config["filename"] == "string_dat_path"
    assert config["data_type"] == np.dtype(np.uint8).name
    assert config["population_size"] == 100
    assert config["genome_length"] == 500
    assert config["filesize"] == 500 * 100 * np.dtype(np.uint8).itemsize


def test_create_json_with_string_datatype(temp_file):

    create_memmap_config_json(
        path=temp_file,
        dat_path="string_dat_path",
        datatype="uint8",
        population_size=100,
        genome_length=500,
    )
    assert temp_file.exists()
    with open(temp_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config["filename"] == "string_dat_path"
    assert config["data_type"] == np.dtype(np.uint8).name
    assert config["population_size"] == 100
    assert config["genome_length"] == 500
    assert config["filesize"] == 500 * 100 * np.dtype(np.uint8).itemsize


def test_create_json_with_non_integer_input(temp_file):
    create_memmap_config_json(
        path=temp_file,
        dat_path=temp_file,
        datatype="uint8",
        population_size=100.0,
        genome_length="500",
    )
    assert temp_file.exists()
    with open(temp_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config["filename"] == str(temp_file)
    assert config["data_type"] == np.dtype(np.uint8).name
    assert config["population_size"] == 100
    assert config["genome_length"] == 500
    assert config["filesize"] == 500 * 100 * np.dtype(np.uint8).itemsize


def test_load_memmap_with_correct_data_and_config(tmp_path):
    data = [100, 10]
    filename_constant = "test_tmp_memmap"
    _test_config_and_mmap_json(tmp_path, data, filename_constant)
    memmap_file, config = load_memmap(
        temp=tmp_path, filename_constant="test_tmp_memmap", open_mode="r"
    )
    assert memmap_file is not None and config is not None
    assert memmap_file.shape == (config["population_size"], config["genome_length"])
    assert (
        Path(tmp_path / f"{filename_constant}.dat").stat().st_size == config["filesize"]
    )


def test_load_memmap_without_filepath(tmp_path):
    data = [100, 100]
    # default filename_constant set for load_memmap is 'population'
    filename_constant = "population"
    _test_config_and_mmap_json(tmp_path, data, filename_constant)
    memmap, config = load_memmap(
        temp=tmp_path,
        # filename is set as none
        filename_constant=None,
        # open mode is not defined
    )
    assert memmap is not None and config is not None
    assert memmap.shape == (config["population_size"], config["genome_length"])
    assert (
        Path(tmp_path / f"{filename_constant}.dat").stat().st_size == config["filesize"]
    )


def test_fail_config_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="population.json does not exist"):
        _ = load_memmap(
            temp=tmp_path,
            filename_constant=None,
        )


def test_fail_config_is_empty(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("")
    with pytest.raises(ValueError, match="config.json is corrupted"):
        _ = load_memmap(temp=tmp_path, filename_constant="config")


def test_fail_config_is_full_whitespaces(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("\n     \n   \n  ")
    with pytest.raises(ValueError, match="config.json is corrupted"):
        _, config = load_memmap(temp=tmp_path, filename_constant="config")


def test_fail_config_exist_no_dat_file(tmp_path):
    _test_config_and_mmap_json(tmp_path, [10, 100], "population")
    dat_path = tmp_path / "population.dat"
    os.remove(dat_path)
    with pytest.raises(FileNotFoundError, match="population.dat does not exist"):
        _ = load_memmap(
            temp=tmp_path,
        )


def test_fail_config_exist_corrupted_dat_file(tmp_path):
    _test_config_and_mmap_json(tmp_path, [10, 100], "population")
    dat_path = tmp_path / "population.dat"
    os.remove(dat_path)
    dat_path.write_text(" ")
    with pytest.raises(ValueError, match="population.dat is corrupted"):
        _ = load_memmap(
            temp=tmp_path,
        )


def _test_config_and_mmap_json(path, data, fname_constant):
    memmap_path = Path(path / f"{fname_constant}.dat")
    config_path = Path(path / f"{fname_constant}.json")
    config = {
        "filename": str(memmap_path),
        "data_type": np.dtype(np.uint8).name,
        "population_size": data[0],
        "genome_length": data[1],
        "filesize": data[0] * data[1] * np.dtype(np.uint8).itemsize,
    }
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    np.memmap(filename=memmap_path, shape=tuple(data), mode="w+")


def test_load_yaml_config_returns_expected_dict(tmp_path):
    yaml_content = {
        "data": {
            "filename": "items.csv",
            "max_weight": 100,
        },
        "population": {
            "size": 50,
            "generations": 200,
            "stream_batch_size": 10,
        },
        "selection": {
            "type": "tournament",
            "selection_pressure": 0.8,
        },
        "genetic_operators": {
            "crossover_type": "one_point",
            "crossover_probability": 0.9,
            "mutation_probability": 0.05,
            "penalty_multiplier": 2.0,
        },
        "experiment": {
            "seed": 42,
            "identifier": "test_experiment",
            "log_level": "INFO",
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(yaml_content, f)
    result = load_yaml_config(config_path)
    expected = {
        "data_filename": "items.csv",
        "max_weight": 100,
        "population_size": 50,
        "generations": 200,
        "stream_batch_size": 10,
        "selection_type": "tournament",
        "selection_pressure": 0.8,
        "crossover_type": "one_point",
        "crossover_probability": 0.9,
        "mutation_probability": 0.05,
        "penalty": 2.0,
        "seed": 42,
        "experiment_identifier": "test_experiment",
        "log_level": "INFO",
    }

    assert result == expected


def test_load_yaml_config_missing_data_section(tmp_path: Path) -> None:
    yaml_content = {
        "data": {
            "filename": "items.csv",
            "max_weight": 100,
        },
        "population": {
            "size": 50,
            "generations": 200,
            "stream_batch_size": 10,
        },
        "selection": {
            "type": "tournament",
            "selection_pressure": 0.8,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(yaml_content, f)
    with pytest.raises(KeyError):
        load_yaml_config(config_path)
