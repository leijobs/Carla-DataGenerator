import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt


def read_json(json_file):
    result_dict = {}
    result_list = []
    with open(json_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                if "odomToCamera" in data:
                    result_dict["odomToCamera"] = data["odomToCamera"]
                    result_list = data["odomToCamera"]
            except json.JSONDecodeError as e:
                    print(f"error {e}")
    return result_dict, result_list


