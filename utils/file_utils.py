import os
import sys
import csv
import json
import h5py
import torch


def load_txt(fn):
    with open(fn, "r") as fh:
        lines = fh.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        lines[i] = lines[i].replace("\r", "")
    return lines


def load_csv(fn, sep=",", header=0, quotechar='"'):
    lines = load_txt(fn)
    csv_reader = csv.reader(
        lines,
        quotechar=quotechar,
        quoting=csv.QUOTE_ALL,
        delimiter=sep,
    )
    data = []
    for i, l in enumerate(csv_reader):
        if i == 0:
            desc = l[:]
            for _ in l:
                data.append([])
        elif len(l) != len(data):
            print("Error reading " + fn)
            sys.exit()
        if i < header:
            continue
        for j, item in enumerate(l):
            data[j].append(item)
    return desc, data, len(data[0])


def load_json(fn):
    with open(fn, "r") as fh:
        d = json.load(fh)
    return d


def load_jsons(fn, limit_lines=None):
    with open(fn, "r") as fh:
        d = []
        for line in fh:
            aux = json.loads(line)
            d.append(aux)
            if limit_lines is not None and len(d) == limit_lines:
                break
    return d

def save_to_hdf5(file_path, data_dict, dataset_dims=None, batch_start=None):
    """
    Appends or initializes datasets in an HDF5 file.
    """
    with h5py.File(file_path, "a") as f:
        for key, value in data_dict.items():
            arr = value.detach().cpu().numpy()
            if key not in f:
                maxshape = list(arr.shape)
                if dataset_dims and key in dataset_dims:
                    maxshape[0] = dataset_dims[key][0]  # override first dim with max
                else:
                    maxshape[0] = None  # unlimited first dimension
                f.create_dataset(
                    key, data=arr, maxshape=tuple(maxshape), chunks=True
                )
            else:
                dset = f[key]
                if batch_start is None:
                    batch_start = dset.shape[0]
                new_size = batch_start + arr.shape[0]
                dset.resize((new_size,) + dset.shape[1:])
                dset[batch_start:new_size] = arr

def load_from_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        query_c = torch.from_numpy(f["clique"][:])
        query_i = torch.from_numpy(f["index"][:])
        query_z = torch.from_numpy(f["z"][:])
        query_m = torch.from_numpy(f["m"][:])
    return query_c, query_i, query_z, query_m
        
